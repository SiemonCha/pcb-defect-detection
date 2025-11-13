# Docker Usage

This directory contains reference containers for running the PCB defect detection stack.

## Build images

```bash
cd docker
docker compose build
```

## Launch the API

```bash
docker compose up api
```

The FastAPI server will be available at `http://localhost:8000`. Logs, runs, and outputs
are mounted from the host.

## Run a training job

```bash
docker compose run --rm trainer pcb-dd train-baseline --epochs 20
```

Adjust the command to invoke other CLI operations. GPU support requires the NVIDIA
Container Toolkit and adding `deploy: resources: reservations: devices` entries to the
compose file.

> Tip: provide a dataset volume (for example, `../datasets`) so both host and containers
> share artifacts.
