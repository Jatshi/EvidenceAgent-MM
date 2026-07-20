FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --uid 10001 eamm
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
USER eamm
EXPOSE 8000
CMD ["eamm", "--db", "/tmp/eamm.db", "serve", "--host", "0.0.0.0", "--port", "8000"]
