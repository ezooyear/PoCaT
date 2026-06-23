# Docker

## Prerequisites

- Docker Desktop or Docker Engine with Compose
- A project `.env` file based on `.env.example`
- Product PDFs under `data/pdfs`

## Run

Build and start PostgreSQL, the MCP server, and the Streamlit app:

```bash
docker compose up --build
```

The Docker image installs CPU-only PyTorch first so the build does not download CUDA packages.
It uses `requirements.docker.txt`, which excludes local-only evaluation/A2A packages from the default app image.

If Docker Desktop drops the build with `failed to receive status` or `error reading from server: EOF`,
restart Docker Desktop and retry with Buildx bake disabled:

```powershell
$env:COMPOSE_BAKE="false"
docker compose build app
docker compose up
```

Open the app at:

```text
http://localhost:8501
```

The MCP server is exposed at:

```text
http://localhost:8000/mcp
```

## Build Vector DB

Run this once after adding or changing PDFs:

```bash
docker compose --profile tools run --rm vectorstore
```

The generated Chroma DB is stored in the `chroma_data` Docker volume and shared with the app container.

## Database

PostgreSQL starts with the schema in `docker/postgres/init/01_schema.sql`.
The database data is stored in the `postgres_data` Docker volume.

To reset Docker-managed data:

```bash
docker compose down -v
```

## Environment Notes

Inside Compose, service hostnames are used instead of localhost:

- `DB_HOST=postgres`
- `MCP_POSTGRES_URL=http://mcp:8000/mcp`

Keep secrets in `.env`; it is ignored by Docker build context and Git.
