FROM python:3.10-slim AS builder
WORKDIR /
COPY ./product/recsys_api/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY ./product/recsys_api /app
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
WORKDIR /app
ENTRYPOINT [ "python3", "main.py" ]


