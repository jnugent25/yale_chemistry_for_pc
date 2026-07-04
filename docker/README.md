# Containerized NMR Tuning on the Cloud

This directory contains containerization assets to build and run the NMR dictionary representation tuning on cloud platforms (e.g., GCP or AWS).

## Files
- `Dockerfile`: Multi-stage build configuration using `uv` for fast dependency caching and a minimal runtime image.
- `.dockerignore`: Excludes local data files (`.pkl`, `.json`), python caches, and virtual environments from the build context.

---

## 1. Local Verification (Local Docker)

First, make sure to build the image from the **repository root directory**:

```bash
# Build the image
docker build -f docker/Dockerfile -t nmr-tuner .

# Verify entrypoint help menu
docker run --rm nmr-tuner --help
```

To run a fast test trial with local files mounted to the container:
```bash
docker run --rm \
  -v /path/to/local/downloads:/data \
  -v /path/to/local/output:/output \
  nmr-tuner \
  tune_representation.py \
    --raw-data /data/ids_nmr_10k.pkl \
    --out /output/ids_nmr_repr_sweep.json \
    --n-trials 2
```

---

## 2. Google Cloud Platform (GCP) Run Guide

Since tuning can take several hours depending on the trials and sample size, we recommend running it as a **Google Cloud Run Job** or a **Vertex AI Custom Job**.

### Build and Push to GCP Artifact Registry
1. **Create an Artifact Registry repository** (replace `<region>` and `<project-id>`):
   ```bash
   gcloud artifacts repositories create nmr-repo \
       --repository-format=docker \
       --location=<region>
   ```
2. **Configure Docker authentication**:
   ```bash
   gcloud auth configure-docker <region>-docker.pkg.dev
   ```
3. **Build and Tag the Image**:
   ```bash
   docker build -f docker/Dockerfile -t <region>-docker.pkg.dev/<project-id>/nmr-repo/nmr-tuner:latest .
   ```
4. **Push the Image**:
   ```bash
   docker push <region>-docker.pkg.dev/<project-id>/nmr-repo/nmr-tuner:latest
   ```

### Option C: Build remotely using Google Cloud Build (No local Docker needed)
If you do not have Docker installed locally on your Mac, you can run the build completely on Google Cloud's servers using the provided `docker/cloudbuild.yaml` file:
```bash
gcloud builds submit \
    --config=docker/cloudbuild.yaml \
    --substitutions=_IMAGE_NAME="<region>-docker.pkg.dev/<project-id>/nmr-repo/nmr-tuner:latest" .
```
*(This command uploads the source code to Cloud Build, executes the build using the remote container engine referencing your custom Dockerfile, and pushes it directly to your Artifact Registry repository.)*

### Option A: Run as a Cloud Run Job (Simple & Cheap)
1. **Create the Job** (providing the command-line arguments to run):
   ```bash
   gcloud run jobs create nmr-tuner-job \
       --image=<region>-docker.pkg.dev/<project-id>/nmr-repo/nmr-tuner:latest \
       --cpu=8 \
       --memory=32Gi \
       --task-timeout=4h \
       --tasks=1 \
       --max-retries=0 \
       --region=<region>
   ```
2. **Execute the Job**:
   ```bash
   gcloud run jobs execute nmr-tuner-job --region=<region>
   ```

### Option B: Run as a Vertex AI Custom Training Job (For GPU/Large Machine scaling)
If you need high CPU counts or memory, Vertex AI custom jobs have no time limit and scale to much larger instances:
```bash
gcloud ai custom-jobs create \
    --region=<region> \
    --display-name=nmr-tuning-custom-job \
    --worker-pool-spec=replica-count=1,machine-type=n1-standard-8,container-image-uri=<region>-docker.pkg.dev/<project-id>/nmr-repo/nmr-tuner:latest \
    --args="tune_representation.py","--raw-data","/gcs/my-bucket/ids_nmr_10k.pkl","--out","/gcs/my-bucket/ids_nmr_repr_sweep.json","--n-trials","50"
```
*(Vertex AI automatically mounts Cloud Storage buckets at `/gcs/bucket-name` if your container reads/writes there.)*

---

## 3. Amazon Web Services (AWS) Run Guide

On AWS, you can run this container using **AWS Batch** or an **ECS Fargate Task**.

### Push to Amazon ECR
1. **Create ECR repository**:
   ```bash
   aws ecr create-repository --repository-name nmr-tuner
   ```
2. **Log in to ECR**:
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
   ```
3. **Build, Tag, and Push**:
   ```bash
   docker build -f docker/Dockerfile -t nmr-tuner .
   docker tag nmr-tuner:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/nmr-tuner:latest
   docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/nmr-tuner:latest
   ```

### Run on ECS Fargate
1. Define a Task Definition in ECS with:
   - Image: `<aws-account-id>.dkr.ecr.<region>.amazonaws.com/nmr-tuner:latest`
   - CPU: `4 vCPU`, Memory: `8 GB` (or larger)
   - Command: `tune_representation.py, --raw-data, /mnt/efs/ids_nmr_10k.pkl, --out, /mnt/efs/ids_nmr_repr_sweep.json, --n-trials, 50`
   - Storage: Mount an EFS volume containing your dataset.
2. Run task using **Fargate launch type** (serverless container execution).
