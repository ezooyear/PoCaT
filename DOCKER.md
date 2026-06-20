# Docker run guide

## 1. Prepare environment variables

Use the existing `.env` file, or copy the Docker example:

```powershell
Copy-Item .env.docker.example .env
```

Then set real values for:

- `OPENROUTER_API_KEY`
- `DB_PASSWORD`
- Optional Langfuse keys

Docker Compose overrides container-only connection values automatically:

- `DB_HOST=postgres`
- `MCP_POSTGRES_URL=http://mcp:8000/mcp`

## 2. Start Docker Desktop

Make sure Docker Desktop is running, then check:

```powershell
docker --version
docker compose version
```

## 3. Build and run

```powershell
docker compose up --build
```

Open the app:

```text
http://localhost:8501
```

The MCP server is exposed at:

```text
http://localhost:8000/mcp
```

PostgreSQL is exposed on:

```text
localhost:5432
```

## 4. Run in the background

```powershell
docker compose up -d --build
```

View logs:

```powershell
docker compose logs -f app
docker compose logs -f mcp
docker compose logs -f postgres
```

Stop everything:

```powershell
docker compose down
```

Stop and remove the PostgreSQL volume too:

```powershell
docker compose down -v
```

## 5. Rebuild Vector DB

The local `data` folder is mounted into the app container. To rebuild ChromaDB from `data/pdfs`:

```powershell
docker compose run --rm app python scripts/build_vectorstore.py
```

## Notes

- The first build can take a while because `torch`, `chromadb`, and `sentence-transformers` are large.
- If the database needs seed data, import it into the `postgres` service after startup or add SQL files under a compose-mounted init directory.
