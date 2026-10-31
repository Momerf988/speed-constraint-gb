# The data pipeline in one image: download DfT's files, load them into Postgres, build the dbt
# layers and export the marts to Parquet. It runs against whichever Postgres the PG* and
# DATABASE_URL environment variables point to, so the same image runs on a laptop, in CI, or
# later as a cloud job. GitHub Actions builds it on every push and publishes it from main.
#
#   docker run --rm -e PGHOST=... -e PGUSER=... -e PGPASSWORD=... -e PGDATABASE=speed_gb \
#       -e DATABASE_URL="host=... user=... password=... dbname=speed_gb" \
#       ghcr.io/momerf988/speed-constraint-gb

FROM python:3.14.3-slim

# Links the published image to its source repository on GitHub
LABEL org.opencontainers.image.source="https://github.com/Momerf988/speed-constraint-gb"

# uv, pinned to the version the project is developed with
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

# make runs the pipeline steps, psql runs the SQL scripts, libgomp is the OpenMP runtime XGBoost needs
RUN apt-get update \
 && apt-get install -y --no-install-recommends make postgresql-client libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_FROZEN=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first: they change rarely, so Docker reuses this layer when only code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY . .
RUN uv sync

CMD ["make", "ingest", "db", "transform", "export"]
