FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap=0.8.0-2+deb12u1 ca-certificates=20250419~deb12u2

RUN uv pip install --system --no-cache \
    "httpx==0.28.1" \
    "pydantic==2.12.5" \
    "numpy==2.2.6" \
    "scipy==1.16.3" \
    "pandas==2.3.3" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "pymc==5.26.1" \
    "arviz==0.23.0"

WORKDIR /app
