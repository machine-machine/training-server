# Training Server Deployment Guide

## Prerequisites on the 3090 Server

```bash
# Verify GPU
nvidia-smi  # Should show 2x RTX 3090

# Verify Docker + Compose + NVIDIA runtime
docker --version          # >= 24.0
docker compose version    # >= 2.20
nvidia-container-cli info # NVIDIA container toolkit installed
```

If NVIDIA container toolkit is missing:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Option A: Deploy with Coolify (Recommended)

### 1. Push code to remote

```bash
cd /path/to/2DEXY
git add training-server/
git push origin 3090-training
```

### 2. In Coolify dashboard

1. Create a new **Docker Compose** resource
2. Point it to this repo, branch `3090-training`, path `training-server/docker-compose.coolify.yml`
3. Set these environment variables in Coolify:

| Variable | Value | Notes |
|----------|-------|-------|
| `POSTGRES_PASSWORD` | *(generate strong password)* | Used by postgres + api + worker |
| `API_KEY` | *(generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)* | Bearer token for all API calls |

That's it — only 2 secrets needed. Everything else is hardcoded or derived in the compose file.

### 3. Verify deployment

```bash
# From the server
curl -s https://ml-api.2dexy.com/health | python3 -m json.tool
curl -s -H "Authorization: Bearer YOUR_API_KEY" https://ml-api.2dexy.com/models/manifest
# Admin UI at https://ml-admin.2dexy.com
```

---

## Option B: Deploy with docker compose directly

### 1. Clone and configure

```bash
ssh your-server
cd /opt/2dexy
git clone https://github.com/YOUR_ORG/2DEXY.git
cd 2DEXY/training-server

# Create .env from template
cp .env.example .env

# Edit secrets (REQUIRED)
nano .env
# Change: POSTGRES_PASSWORD, API_KEY, CODE_SERVER_PASSWORD
```

### 2. Start the stack

```bash
docker compose up -d --build
```

### 3. Verify

```bash
# Check all 6 services running
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check GPU visibility in worker
docker compose exec celery-worker nvidia-smi

# Check DB migrations ran
docker compose logs api | grep "alembic"

# Admin UI
open http://localhost:8501
```

### 4. Test a training job

```bash
# Upload a dataset
curl -X POST http://localhost:8000/data/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@sample_data.csv" \
  -F "name=test-dataset" \
  -F "description=Test upload"

# Launch ensemble training
curl -X POST http://localhost:8000/jobs/train \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "ensemble",
    "dataset_id": "DATASET_ID_FROM_UPLOAD",
    "hyperparams": {}
  }'

# Check job status
curl http://localhost:8000/jobs/JOB_ID/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Service Architecture

```
                    ┌─────────────────┐
                    │  ml-admin.2dexy │ :8501
                    │  (Streamlit UI) │
                    └────────┬────────┘
                             │ HTTP
                    ┌────────▼────────┐
                    │  ml-api.2dexy   │ :8000
   Internet ───────►  (FastAPI)       │
                    └───┬────────┬────┘
                        │        │
              ┌─────────▼─┐  ┌──▼──────────────┐
              │  Postgres  │  │  Redis           │
              │  :5432     │  │  :6379           │
              └────────────┘  └──┬───────────────┘
                                 │ Celery queue
                        ┌────────▼────────┐
                        │  GPU Worker     │
                        │  2x RTX 3090    │
                        │  shm_size=32gb  │
                        └─────────────────┘
```

## Ports (Dev Compose)

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000/docs |
| Admin UI | 8501 | http://localhost:8501 |
| Code Server | 8443 | http://localhost:8443 |
| Postgres | 5432 | Internal only |
| Redis | 6379 | Internal only |

## Ports (Coolify — via Traefik)

| Service | Domain |
|---------|--------|
| API | https://ml-api.2dexy.com |
| Admin UI | https://ml-admin.2dexy.com |

---

## Troubleshooting

### GPU not visible in worker
```bash
docker compose exec celery-worker nvidia-smi
# If "command not found": NVIDIA container toolkit not installed
# If "no devices": Check CUDA_VISIBLE_DEVICES env var
```

### Celery worker won't start
```bash
docker compose logs celery-worker
# Common: ImportError — rebuild with: docker compose build celery-worker
```

### API returns 500 on startup
```bash
docker compose logs api | head -50
# Common: Alembic migration failed — check DATABASE_URL matches postgres credentials
```

### Admin UI can't reach API
```bash
# Dev: API_URL should be http://api:8000 (Docker internal DNS)
# Coolify: API_URL should be http://training-api:8000
```
