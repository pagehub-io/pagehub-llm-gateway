FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY api ./api
RUN pip install --no-cache-dir -e .

ARG GIT_COMMIT=local-dev
ENV GIT_COMMIT=$GIT_COMMIT
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "4011"]
