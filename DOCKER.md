# Docker 실행 가이드

이 문서는 PoCaT 프로젝트를 Docker로 실행하는 방법을 설명합니다.

Docker를 사용하면 로컬 PC에 Python 패키지, PostgreSQL, MCP 서버 실행 환경을 각각 따로 맞추지 않아도 됩니다. `docker compose up` 명령으로 Streamlit 앱, PostgreSQL, MCP 서버를 함께 실행할 수 있습니다.

## 1. Docker로 실행하면 달라지는 점

로컬 실행은 보통 다음처럼 직접 앱을 실행합니다.

```powershell
streamlit run app.py
```

Docker 실행은 다음처럼 여러 서비스를 컨테이너로 함께 실행합니다.

```powershell
docker compose up
```

이 프로젝트의 Docker Compose 구성은 다음 서비스를 띄웁니다.

| 서비스 | 역할 | 컨테이너 이름 | 외부 접속 |
| --- | --- | --- | --- |
| `app` | Streamlit 웹 앱 | `pocat-app` | `http://localhost:8501` |
| `mcp` | PostgreSQL MCP 서버 | `pocat-mcp` | `http://localhost:8000/mcp` |
| `postgres` | PostgreSQL 데이터베이스 | `pocat-postgres` | `localhost:5432` |

컨테이너끼리는 Docker 내부 네트워크에서 서비스 이름으로 통신합니다.

- 앱 컨테이너에서 DB 접속 주소: `postgres`
- 앱 컨테이너에서 MCP 서버 주소: `http://mcp:8000/mcp`
- 브라우저에서 앱 접속 주소: `http://localhost:8501`

즉, 컨테이너 안에서는 `localhost`가 내 PC가 아니라 해당 컨테이너 자신을 의미합니다. 그래서 Compose에서는 DB 주소를 `localhost`가 아닌 `postgres`로 지정합니다.

## 2. 사전 준비

Docker Desktop이 실행 중인지 확인합니다.

```powershell
docker --version
docker compose version
```

두 명령이 정상적으로 버전을 출력하면 Docker를 사용할 준비가 된 상태입니다.

## 3. 환경 변수 준비

프로젝트 루트에 `.env` 파일이 필요합니다.

이미 `.env`가 있다면 그대로 사용할 수 있습니다. 새로 만들려면 Docker 예시 파일을 복사합니다.

```powershell
Copy-Item .env.docker.example .env
```

그다음 `.env`에서 실제 값을 채웁니다.

필수로 확인할 값:

- `OPENROUTER_API_KEY`: OpenRouter API 키
- `DB_PASSWORD`: PostgreSQL 비밀번호

선택 값:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

Docker Compose는 컨테이너 내부 연결에 필요한 일부 값을 자동으로 덮어씁니다.

```yaml
DB_HOST: postgres
DB_PORT: 5432
MCP_POSTGRES_URL: http://mcp:8000/mcp
```

따라서 Docker로 실행할 때는 `.env`의 `DB_HOST`가 로컬용 값이어도 Compose 실행 중에는 컨테이너용 값이 적용됩니다.

## 4. 빌드하고 실행하기

처음 실행하거나 Dockerfile, requirements, 소스 코드가 크게 바뀐 경우에는 빌드와 실행을 함께 합니다.

```powershell
docker compose up --build
```

실행이 완료되면 브라우저에서 앱을 엽니다.

```text
http://localhost:8501
```

MCP 서버는 다음 주소로 노출됩니다.

```text
http://localhost:8000/mcp
```

PostgreSQL은 로컬 PC에서 다음 주소로 접근할 수 있습니다.

```text
localhost:5432
```

## 5. 백그라운드 실행

터미널을 계속 점유하지 않고 뒤에서 실행하려면 `-d` 옵션을 사용합니다.

```powershell
docker compose up -d --build
```

실행 중인 컨테이너 상태를 확인합니다.

```powershell
docker compose ps
```

로그를 확인합니다.

```powershell
docker compose logs -f app
docker compose logs -f mcp
docker compose logs -f postgres
```

전체 로그를 한 번에 보려면 다음 명령을 사용합니다.

```powershell
docker compose logs -f
```

## 6. 종료하기

컨테이너를 종료하고 네트워크를 정리합니다.

```powershell
docker compose down
```

이 명령은 PostgreSQL 데이터 볼륨을 삭제하지 않습니다. 따라서 DB 데이터는 유지됩니다.

PostgreSQL 데이터까지 완전히 삭제하려면 `-v` 옵션을 사용합니다.

```powershell
docker compose down -v
```

주의: `docker compose down -v`를 실행하면 `postgres_data` 볼륨이 삭제되어 DB 데이터가 사라집니다.

## 7. 데이터와 볼륨

현재 Compose 설정에서는 로컬 `data` 폴더가 앱 컨테이너의 `/app/data`에 연결됩니다.

```yaml
volumes:
  - ./data:/app/data
```

그래서 로컬의 `data` 폴더에 있는 PDF, ChromaDB 데이터, 기타 파일은 컨테이너에서도 사용할 수 있습니다.

PostgreSQL 데이터는 Docker volume인 `postgres_data`에 저장됩니다.

```yaml
volumes:
  postgres_data:
```

이 볼륨은 `docker compose down`으로는 삭제되지 않고, `docker compose down -v`를 실행해야 삭제됩니다.

## 8. Vector DB 재생성

로컬 `data/pdfs` 폴더의 PDF를 기준으로 ChromaDB를 다시 만들려면 다음 명령을 실행합니다.

```powershell
docker compose run --rm app python scripts/build_vectorstore.py
```

이 명령은 임시 `app` 컨테이너를 실행해서 벡터 DB 생성 스크립트를 수행한 뒤 컨테이너를 제거합니다.

## 9. 자주 쓰는 명령어

이미지를 다시 빌드하고 실행:

```powershell
docker compose up --build
```

백그라운드에서 빌드 후 실행:

```powershell
docker compose up -d --build
```

실행 중인 서비스 확인:

```powershell
docker compose ps
```

앱 로그 확인:

```powershell
docker compose logs -f app
```

MCP 서버 로그 확인:

```powershell
docker compose logs -f mcp
```

PostgreSQL 로그 확인:

```powershell
docker compose logs -f postgres
```

컨테이너 종료:

```powershell
docker compose down
```

컨테이너와 DB 볼륨까지 삭제:

```powershell
docker compose down -v
```

이미지만 빌드:

```powershell
docker compose build
```

앱 컨테이너 안에서 명령 실행:

```powershell
docker compose run --rm app python scripts/build_vectorstore.py
```

## 10. 문제 해결

### 포트가 이미 사용 중인 경우

다음과 비슷한 오류가 나면 해당 포트를 다른 프로그램이 사용 중인 상태입니다.

```text
port is already allocated
```

확인할 포트:

- 앱: `8501`
- MCP 서버: `8000`
- PostgreSQL: `5432`

이미 로컬 PostgreSQL이 `5432` 포트를 사용 중이면 `docker-compose.yml` 또는 `.env`의 `DB_PORT`를 다른 값으로 바꿀 수 있습니다.

예:

```env
DB_PORT=5433
```

이 경우 로컬 PC에서는 `localhost:5433`으로 PostgreSQL 컨테이너에 접근합니다. 컨테이너 내부 통신은 그대로 `postgres:5432`를 사용합니다.

### API 키 오류가 나는 경우

`.env`의 `OPENROUTER_API_KEY` 값이 실제 키인지 확인합니다.

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

예시 값이 그대로 남아 있으면 LLM 호출이 실패합니다.

### DB 연결 오류가 나는 경우

Compose 실행 중에는 앱과 MCP 서버가 DB에 접근할 때 `DB_HOST=postgres`를 사용해야 합니다.

`docker-compose.yml`에는 이 값이 이미 지정되어 있습니다.

```yaml
environment:
  DB_HOST: postgres
  DB_PORT: 5432
```

직접 `docker run`으로 실행하는 경우에는 이 네트워크 설정이 자동으로 잡히지 않으므로 Compose 사용을 권장합니다.

### 첫 빌드가 오래 걸리는 경우

첫 빌드는 오래 걸릴 수 있습니다. 이 프로젝트는 다음과 같은 무거운 Python 패키지를 설치합니다.

- `torch`
- `chromadb`
- `sentence-transformers`

한 번 빌드된 레이어는 Docker 캐시에 저장되므로, 이후 빌드는 보통 더 빠릅니다.

## 11. 로컬 실행과 Docker 실행 중 무엇을 쓰면 좋을까?

빠르게 코드 수정하면서 개발할 때:

```powershell
streamlit run app.py
```

DB, MCP 서버까지 한 번에 같은 환경으로 띄우고 싶을 때:

```powershell
docker compose up --build
```

팀원이나 다른 PC에서도 같은 방식으로 실행되게 만들고 싶을 때는 Docker 실행을 권장합니다.
