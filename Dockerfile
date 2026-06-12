FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    GYMPULSE_CSV=/data/field_dispatch_dataa.csv

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY pangu_dispatcher_core/ ./pangu_dispatcher_core/
COPY field_dispatch_dataa.csv /data/field_dispatch_dataa.csv

RUN useradd --create-home --uid 10001 gympulse \
    && mkdir -p /output \
    && chown -R gympulse:gympulse /app /data /output

USER gympulse

VOLUME ["/data", "/output"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["gunicorn", "--chdir", "/app/pangu_dispatcher_core", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1", "--timeout", "120", "web_app:app"]
