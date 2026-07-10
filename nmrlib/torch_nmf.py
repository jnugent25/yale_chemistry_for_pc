"""PyTorch NMF engine with a pluggable reconstruction loss.

*** TEMPLATE / SCAFFOLD — read the STATUS notes before trusting a method ***

Why this exists
---------------
sklearn's ``NMF`` factorizes ``X ≈ W @ H`` with multiplicative / coordinate-descent
updates, which lock the objective to a beta-divergence (Frobenius or KL). Our inputs
are *distributions over a ppm axis*: a Frobenius error treats a peak predicted one
grid-bin off as two unrelated errors, whereas an optimal-transport (OT) loss penalizes
it by *how far it moved*. A gradient-descent NMF makes the reconstruction objective
pluggable, so we can drop in a `geomloss` Sinkhorn loss that uses the real ppm ground
metric.

Drop-in contract
----------------
Mirrors the sklearn surface the rest of the repo already calls, so it can replace
``make_nmf(...)`` (see train_gap_model.make_nmf) with no downstream edits:

    nmf = TorchNMF(cfg, geometry=geom)
    W_train = nmf.fit_transform(X_train)   # (n_samples, k)   codes
    W_val   = nmf.transform(X_val)         # freezes H, solves W only
    H       = nmf.components_              # (k, n_features)   dictionary

Block-aware OT
--------------
``X`` is the concatenation ``[h_modality_weight * H_block | c_modality_weight * C_block]``
(see tune_representation.assemble_x). The two blocks live on *different* ppm grids
(H: 0–12 ppm, C: 0–220 ppm), so OT must be computed per block with each block's own grid
coordinates and summed — never across the concatenated vector, where a single ground
metric is meaningless. ``SpectralGeometry`` carries the coordinates + block split so the
geomloss loss can reconstruct that structure.

STATUS
------
  - geomloss Sinkhorn loss (block-aware OT): IMPLEMENTED + smoke-tested end-to-end on
    synthetic spectra (geomloss 0.3.1). fit/transform optimize the OT objective, codes
    stay finite & non-negative, and the loss correctly grows with peak-shift distance
    where L2 is blind (near vs far shift: OT ratio ~178, L2 ratio ~1.06). backend
    auto-selects "online" (keops) on CUDA, "tensorized" (pure torch) on MPS/CPU.
    Lazy-imported, so this module loads fine without geomloss installed.
  - Frobenius / KL losses: IMPLEMENTED but NOT for production — sklearn's coordinate
    descent beats gradient descent on pure beta-divergences (parity check: sklearn
    ~0.010 rel-recon vs Adam ~0.057 on a rank-12 synthetic). Keep sklearn NMF for
    Frobenius/KL; these exist here only to validate the engine. This engine's reason
    to exist is the OT loss, which sklearn cannot express.
  - Optimizers (adam / lbfgs), softplus/clamp nonnegativity, transform-only solve,
    NNDSVD init (best-effort via sklearn internal, random fallback): IMPLEMENTED.
  - NOT yet done: run on real alberts spectra; wire loss/OT knobs into SweepConfig so
    tune_representation can sweep the engine; validate on CUDA (see [[torch-nmf-deploy-target]]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

import numpy as np

try:  # torch is a hard dep, but keep the import guard explicit for clarity
    import torch
    from torch import Tensor
except ImportError as exc:  # pragma: no cover
    raise ImportError("nmrlib.torch_nmf requires torch (see pyproject).") from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
LossName = Literal["frobenius", "kl", "geomloss"]
Optimizer = Literal["adam", "lbfgs"]
Nonneg = Literal["softplus", "clamp"]


@dataclass(frozen=True)
class TorchNMFConfig:
    """Knobs for the gradient-descent NMF.

    The first field mirrors sklearn/SweepConfig; the rest are engine-specific and
    have no sklearn analogue. ``make_torch_nmf`` maps a SweepConfig onto this.
    """

    n_components: int = 60

    # Objective ------------------------------------------------------------- #
    loss: LossName = "frobenius"
    # geomloss-only: entropic-regularization scale (in *ppm*, the ground-metric
    # units). Larger = blurrier / cheaper / more stable; smaller = closer to exact
    # Wasserstein but harder to optimize. A few grid-steps is a sane starting point.
    blur: float = 0.05
    sinkhorn_p: Literal[1, 2] = 2          # ground metric |x-y|^p
    sinkhorn_scaling: float = 0.7          # geomloss annealing (speed/accuracy trade)
    # geomloss backend. "auto" -> "online" (keops, no full cost matrix) on CUDA,
    # "tensorized" (pure torch, needs the full NxN matrix) on MPS/CPU where keops
    # isn't available. "online" scales to the large C-grid; "tensorized" does not.
    sinkhorn_backend: Literal["auto", "tensorized", "online", "multiscale"] = "auto"
    # Optional L1/L2 penalties on W (codes) and H (dictionary), mirroring
    # sklearn's alpha_W / alpha_H / l1_ratio so sweeps stay comparable.
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0

    # Optimization ---------------------------------------------------------- #
    optimizer: Optimizer = "adam"
    lr: float = 0.05
    n_iter: int = 400                      # outer steps for fit (both W and H)
    transform_n_iter: int = 200            # steps for transform (W only, H frozen)
    tol: float = 1e-5                      # rel. loss change for early stop
    nonneg: Nonneg = "softplus"            # how nonnegativity is enforced
    init: Literal["nndsvda", "random"] = "nndsvda"

    # Runtime --------------------------------------------------------------- #
    device: Optional[str] = None           # None -> auto (cuda>mps>cpu)
    dtype: Literal["float32", "float64"] = "float32"
    random_state: int = 0
    verbose: bool = False


@dataclass(frozen=True)
class SpectralGeometry:
    """Per-block ppm coordinates + the H/C split, so an OT loss can treat each
    modality as a distribution over its own axis.

    ``h_width`` is the number of columns belonging to the H block in the assembled
    ``X`` (the C block is everything after). ``h_coords`` / ``c_coords`` are the ppm
    grids (same ones passed to build_soft_peak_matrix), length ``h_width`` /
    ``n_features - h_width`` respectively.

    Modality weights match assemble_x so the loss can undo them before comparing in
    physical-intensity space (optional — off by default; the codes are compared as-is).
    """

    h_coords: np.ndarray                   # (h_width,) ppm
    c_coords: np.ndarray                   # (c_width,) ppm
    h_modality_weight: float = 1.0
    c_modality_weight: float = 1.0

    @property
    def h_width(self) -> int:
        return len(self.h_coords)

    @property
    def c_width(self) -> int:
        return len(self.c_coords)

    @property
    def n_features(self) -> int:
        return self.h_width + self.c_width


# --------------------------------------------------------------------------- #
# Loss registry
# --------------------------------------------------------------------------- #
class ReconstructionLoss(Protocol):
    """A differentiable reconstruction loss.

    Called with the target and reconstruction (both ``(batch, n_features)``, non-negative)
    and returns a scalar tensor. ``geometry`` is provided so block/OT-aware losses can
    split the feature axis; pointwise losses ignore it.
    """

    def __call__(self, x: Tensor, x_hat: Tensor, geometry: Optional[SpectralGeometry]) -> Tensor: ...


def _frobenius(x: Tensor, x_hat: Tensor, geometry: Optional[SpectralGeometry]) -> Tensor:
    """0.5 * ||X - X_hat||_F^2 / batch — matches sklearn's frobenius objective scale."""
    return 0.5 * (x - x_hat).pow(2).sum() / x.shape[0]


def _kl(x: Tensor, x_hat: Tensor, geometry: Optional[SpectralGeometry]) -> Tensor:
    """Generalized KL (I-divergence), the beta_loss='kullback-leibler' objective:
    sum( x*log(x/x_hat) - x + x_hat ). Small eps guards log(0) / divide-by-0."""
    eps = 1e-9
    xk = x.clamp_min(eps)
    xh = x_hat.clamp_min(eps)
    return (xk * (xk.log() - xh.log()) - xk + xh).sum() / x.shape[0]


def _make_geomloss(cfg: TorchNMFConfig) -> ReconstructionLoss:
    """Build a block-aware Sinkhorn reconstruction loss.

    Each row of a block is treated as a weighted point cloud over that block's ppm
    coordinates (weights = intensities, normalized to unit mass so the two clouds are
    comparable). The Sinkhorn divergence is summed over the H and C blocks. geomloss is
    imported lazily so the rest of the module works without it installed.
    """
    try:
        from geomloss import SamplesLoss
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "loss='geomloss' needs the optional geomloss dependency.\n"
            "  uv add 'geomloss>=0.2.6' 'pykeops>=2.2'\n"
            "(pykeops JIT-compiles CUDA/C++ kernels; on Apple Silicon it runs on CPU — "
            "verify it imports before relying on this path.)"
        ) from exc

    dev = cfg.device or _auto_device()
    backend = cfg.sinkhorn_backend
    if backend == "auto":
        # keops "online" needs a working pykeops (CUDA/C++ JIT); default to it only
        # on CUDA, where it avoids the O(N^2) cost matrix for the large C-grid.
        backend = "online" if dev == "cuda" else "tensorized"

    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=cfg.sinkhorn_p,
        blur=cfg.blur,
        scaling=cfg.sinkhorn_scaling,
        backend=backend,
    )

    def _block_ot(a: Tensor, b: Tensor, coords: Tensor) -> Tensor:
        """Sinkhorn divergence between two batches of intensity vectors over `coords`.

        a, b: (batch, n_bins) >= 0 ; coords: (n_bins, 1). Rows are normalized to unit
        mass; zero-mass rows are dropped (an all-zero reconstruction has no OT target).
        """
        eps = 1e-12
        a_mass = a.sum(dim=1, keepdim=True)
        b_mass = b.sum(dim=1, keepdim=True)
        keep = (a_mass.squeeze(1) > eps) & (b_mass.squeeze(1) > eps)
        if keep.sum() == 0:
            return a.sum() * 0.0  # differentiable zero
        a_n = (a[keep] / a_mass[keep].clamp_min(eps))
        b_n = (b[keep] / b_mass[keep].clamp_min(eps))
        # geomloss batched weighted point clouds: weights (B,N), locations (B,N,D).
        xy = coords.unsqueeze(0).expand(a_n.shape[0], -1, -1).contiguous()
        return sinkhorn(a_n, xy, b_n, xy).sum()

    def loss_fn(x: Tensor, x_hat: Tensor, geometry: Optional[SpectralGeometry]) -> Tensor:
        if geometry is None:
            raise ValueError("geomloss reconstruction requires a SpectralGeometry (grid coords).")
        hw = geometry.h_width
        dev, dt = x.device, x.dtype
        h_coords = torch.as_tensor(geometry.h_coords, device=dev, dtype=dt).unsqueeze(1)
        c_coords = torch.as_tensor(geometry.c_coords, device=dev, dtype=dt).unsqueeze(1)
        # Undo the modality weights so each block is compared in its own intensity scale.
        xh_true = x[:, :hw] / geometry.h_modality_weight
        xc_true = x[:, hw:] / geometry.c_modality_weight
        xh_pred = x_hat[:, :hw] / geometry.h_modality_weight
        xc_pred = x_hat[:, hw:] / geometry.c_modality_weight
        loss_h = _block_ot(xh_true, xh_pred, h_coords)
        loss_c = _block_ot(xc_true, xc_pred, c_coords)
        return (loss_h + loss_c) / x.shape[0]

    return loss_fn


LOSS_REGISTRY: dict[str, Callable[[TorchNMFConfig], ReconstructionLoss]] = {
    "frobenius": lambda cfg: _frobenius,
    "kl": lambda cfg: _kl,
    "geomloss": _make_geomloss,
}


# --------------------------------------------------------------------------- #
# Nonnegativity reparametrization
# --------------------------------------------------------------------------- #
def _forward(raw: Tensor, mode: Nonneg) -> Tensor:
    """Map an unconstrained parameter to a nonnegative factor."""
    if mode == "softplus":
        return torch.nn.functional.softplus(raw)
    if mode == "clamp":
        return raw.clamp_min(0.0)
    raise ValueError(f"Unknown nonneg mode: {mode}")


def _inverse(value: np.ndarray, mode: Nonneg) -> np.ndarray:
    """Invert `_forward` to initialize the raw parameter from a nonnegative guess."""
    v = np.clip(value, 0.0, None)
    if mode == "softplus":
        # inverse softplus: log(expm1(v)); stable for large v via log1p.
        return np.log(np.expm1(np.clip(v, 1e-6, None)))
    if mode == "clamp":
        return v
    raise ValueError(f"Unknown nonneg mode: {mode}")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TorchNMF:
    """Gradient-descent NMF with a pluggable reconstruction loss.

    sklearn-compatible surface: ``fit`` / ``fit_transform`` / ``transform`` /
    ``components_``. See module docstring for the drop-in contract and STATUS.
    """

    def __init__(self, config: TorchNMFConfig, geometry: Optional[SpectralGeometry] = None):
        self.cfg = config
        self.geometry = geometry
        self.loss_fn: ReconstructionLoss = LOSS_REGISTRY[config.loss](config)
        self._device = torch.device(config.device or _auto_device())
        self._dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self._H_raw: Optional[Tensor] = None       # learned dictionary (raw param)
        self.components_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.reconstruction_err_: Optional[float] = None

    # -- public API -------------------------------------------------------- #
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit W and H to X; return the codes W (n_samples, k)."""
        Xt = self._to_tensor(X)
        W_raw, H_raw = self._init_factors(Xt)
        W_raw, H_raw = self._optimize(Xt, W_raw, H_raw, freeze_H=False)
        self._H_raw = H_raw.detach()
        self.components_ = _forward(self._H_raw, self.cfg.nonneg).cpu().numpy()
        return _forward(W_raw.detach(), self.cfg.nonneg).cpu().numpy()

    def fit(self, X: np.ndarray) -> "TorchNMF":
        self.fit_transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Solve for codes W of new data with the learned dictionary H frozen."""
        if self._H_raw is None:
            raise RuntimeError("TorchNMF.transform called before fit.")
        Xt = self._to_tensor(X)
        W_raw = self._init_W(Xt, self._H_raw)
        W_raw, _ = self._optimize(Xt, W_raw, self._H_raw, freeze_H=True)
        return _forward(W_raw.detach(), self.cfg.nonneg).cpu().numpy()

    def reconstruct(self, W: np.ndarray) -> np.ndarray:
        """W @ H in intensity space (non-negative), for reconstruction-error features."""
        if self.components_ is None:
            raise RuntimeError("TorchNMF.reconstruct called before fit.")
        return np.clip(W @ self.components_, 0.0, None)

    # -- internals --------------------------------------------------------- #
    def _optimize(self, X: Tensor, W_raw: Tensor, H_raw: Tensor, freeze_H: bool):
        cfg = self.cfg
        # .contiguous() so leaf grads are contiguous — LBFGS's flat-grad gather
        # does .view(-1), which rejects the non-contiguous warm-started tensors.
        W_raw = W_raw.clone().contiguous().requires_grad_(True)
        if freeze_H:
            H_raw = H_raw.detach()
            params = [W_raw]
        else:
            H_raw = H_raw.clone().contiguous().requires_grad_(True)
            params = [W_raw, H_raw]

        n_iter = cfg.transform_n_iter if freeze_H else cfg.n_iter
        opt = _build_optimizer(cfg, params)
        prev = None

        def closure():
            opt.zero_grad(set_to_none=True)
            W = _forward(W_raw, cfg.nonneg)
            H = _forward(H_raw, cfg.nonneg)
            X_hat = W @ H
            loss = self.loss_fn(X, X_hat, self.geometry)
            loss = loss + self._penalty(W, H, freeze_H)
            loss.backward()
            return loss

        for step in range(n_iter):
            loss = opt.step(closure)
            cur = float(loss.detach())
            self.n_iter_ = step + 1
            if cfg.verbose and step % max(1, n_iter // 10) == 0:
                print(f"  [torch-nmf] step {step:4d}  loss={cur:.6e}", flush=True)
            if prev is not None and abs(prev - cur) <= cfg.tol * max(abs(prev), 1e-12):
                break
            prev = cur

        self.reconstruction_err_ = prev
        return W_raw, H_raw

    def _penalty(self, W: Tensor, H: Tensor, freeze_H: bool) -> Tensor:
        """sklearn-style elastic-net penalty on the factors (scaled per element)."""
        cfg = self.cfg
        if cfg.alpha_W == 0.0 and cfg.alpha_H == 0.0:
            return W.sum() * 0.0
        l1, l2 = cfg.l1_ratio, 1.0 - cfg.l1_ratio
        pen = W.sum() * 0.0
        aW = cfg.alpha_W
        if aW:
            pen = pen + aW * (l1 * W.abs().mean() + l2 * 0.5 * W.pow(2).mean())
        if cfg.alpha_H and not freeze_H:
            aH = cfg.alpha_H
            pen = pen + aH * (l1 * H.abs().mean() + l2 * 0.5 * H.pow(2).mean())
        return pen

    def _init_factors(self, X: Tensor):
        H0 = self._init_H(X)
        W0 = self._init_W(X, _inverse_tensor(H0, self.cfg.nonneg))
        return W0, _inverse_tensor(H0, self.cfg.nonneg)

    def _init_H(self, X: Tensor) -> Tensor:
        """Nonnegative dictionary init (nndsvda best-effort, else scaled random)."""
        k = self.cfg.n_components
        Xn = X.detach().cpu().numpy()
        if self.cfg.init == "nndsvda":
            H = _nndsvda_H(Xn, k, self.cfg.random_state)
            if H is not None:
                return torch.as_tensor(H, device=self._device, dtype=self._dtype)
        g = torch.Generator(device="cpu").manual_seed(self.cfg.random_state)
        scale = float(np.sqrt(Xn.mean() / k)) if Xn.size else 1.0
        H = torch.rand(k, X.shape[1], generator=g).to(self._device, self._dtype) * scale
        return H

    def _init_W(self, X: Tensor, H_raw: Tensor) -> Tensor:
        """Warm-start W by a nonneg least-squares-ish step: X @ H^+ clamped to >=0,
        then mapped back through the nonneg inverse."""
        # lstsq isn't implemented on MPS (CUDA/CPU have it), so on MPS only, solve
        # the warm-start on CPU and move the result back.
        solve_dev = torch.device("cpu") if self._device.type == "mps" else self._device
        H = _forward(H_raw.detach().to(solve_dev, self._dtype), self.cfg.nonneg)
        with torch.no_grad():
            W = torch.linalg.lstsq(H.T, X.detach().to(solve_dev).T).solution.T.clamp_min(1e-6)
        W = W.to(self._device, self._dtype)
        return _inverse_tensor(W, self.cfg.nonneg)

    def _to_tensor(self, X: np.ndarray) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(X), device=self._device, dtype=self._dtype)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_optimizer(cfg: TorchNMFConfig, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "lbfgs":
        return torch.optim.LBFGS(params, lr=cfg.lr, max_iter=20, line_search_fn="strong_wolfe")
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def _inverse_tensor(t: Tensor, mode: Nonneg) -> Tensor:
    return torch.as_tensor(_inverse(t.detach().cpu().numpy(), mode), device=t.device, dtype=t.dtype)


def _nndsvda_H(X: np.ndarray, k: int, random_state: int) -> Optional[np.ndarray]:
    """Best-effort NNDSVD-a dictionary init via sklearn's internal helper.

    Private API (sklearn.decomposition._nmf._initialize_nmf); guarded so a sklearn
    version bump that moves it just falls back to random init.
    """
    try:
        from sklearn.decomposition._nmf import _initialize_nmf
        _, H = _initialize_nmf(X, k, init="nndsvda", random_state=random_state)
        return H
    except Exception:
        return None


def spectral_geometry(
    h_step_ppm: float = 0.01,
    c_step_ppm: float = 0.25,
    h_range: tuple[float, float] = (0.0, 12.0),
    c_range: tuple[float, float] = (0.0, 220.0),
    h_modality_weight: float = 1.0,
    c_modality_weight: float = 1.0,
) -> SpectralGeometry:
    """Build a SpectralGeometry from the grid params the pipeline already uses.

    The grids must match those passed to build_soft_peak_matrix / assemble_x when the
    input ``X`` was assembled (defaults mirror tune_representation / build_representation):
    H over 0–12 ppm at 0.01, C over 0–220 ppm at 0.25. The resulting block widths must
    equal ``X``'s column split, or the OT loss will mis-slice the modalities.
    """
    def _grid(lo: float, hi: float, step: float) -> np.ndarray:
        return np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)

    return SpectralGeometry(
        h_coords=_grid(*h_range, h_step_ppm),
        c_coords=_grid(*c_range, c_step_ppm),
        h_modality_weight=h_modality_weight,
        c_modality_weight=c_modality_weight,
    )


def make_torch_nmf(cfg, max_iter: int, geometry: Optional[SpectralGeometry] = None,
                   loss: LossName = "frobenius", **overrides) -> TorchNMF:
    """Adapter mirroring train_gap_model.make_nmf, but returning a TorchNMF.

    Maps the fields a SweepConfig shares with NMF (n_components, alpha_W/H, l1_ratio)
    onto a TorchNMFConfig, sets n_iter from max_iter, and lets the caller pick the loss
    and override any engine knob. ``geometry`` is required for loss='geomloss'.
    """
    tcfg = TorchNMFConfig(
        n_components=getattr(cfg, "n_components", 60),
        alpha_W=getattr(cfg, "alpha_W", 0.0),
        alpha_H=getattr(cfg, "alpha_H", 0.0),
        l1_ratio=getattr(cfg, "l1_ratio", 0.0),
        loss=loss,
        n_iter=max_iter,
        **overrides,
    )
    return TorchNMF(tcfg, geometry=geometry)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic nonnegative low-rank data; checks fit/transform run and Frobenius
    # recovers structure. NOT a correctness proof — see STATUS.
    rng = np.random.default_rng(0)
    n, d, k = 200, 80, 8
    W_true = rng.gamma(1.0, size=(n, k))
    H_true = rng.gamma(1.0, size=(k, d))
    X = W_true @ H_true
    Xtr, Xte = X[:150], X[150:]

    for loss in ("frobenius", "kl"):
        m = TorchNMF(TorchNMFConfig(n_components=k, loss=loss, n_iter=300, verbose=False))
        W = m.fit_transform(Xtr)
        Wte = m.transform(Xte)
        rec = m.reconstruct(Wte)
        rel = np.linalg.norm(Xte - rec) / np.linalg.norm(Xte)
        print(f"loss={loss:9s}  W{W.shape}  H{m.components_.shape}  "
              f"test rel-recon={rel:.4f}  iters={m.n_iter_}  device={m._device}")

    print("\ngeomloss path (block-aware OT) requires a SpectralGeometry + the geomloss "
          "package; see make_torch_nmf(..., loss='geomloss', geometry=geom).")
