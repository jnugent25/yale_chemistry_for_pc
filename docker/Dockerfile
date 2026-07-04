# Use astral-sh/uv as the builder image for fast, cached builds
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency definition files from build context root
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (frozen lockfile)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Final runtime image
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code files from build context root
COPY *.py ./

# Define entrypoint to python
ENTRYPOINT ["python"]
CMD ["tune_representation.py", "--help"]
