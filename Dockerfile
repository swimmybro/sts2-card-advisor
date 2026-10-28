# Card advisor API. Models are not baked in; mount them at /app/models:
#
#   docker build -t sts2-card-advisor .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" sts2-card-advisor
#
# /app/models/card_pick is required; /app/models/win_prob is optional.

FROM python:3.12-slim-bookworm

# LightGBM's Linux wheels need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models/card_pick \
    WIN_MODEL_DIR=/app/models/win_prob \
    CARD_META=/app/configs/cards.csv

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY configs/cards.csv configs/cards.csv

RUN useradd --create-home --uid 10001 advisor
USER advisor

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "--factory", "sts2_advisor.serve:create_app", "--host", "0.0.0.0", "--port", "8000"]
