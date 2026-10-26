FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app

# XGBoost requires the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["/app/.venv/bin/gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
