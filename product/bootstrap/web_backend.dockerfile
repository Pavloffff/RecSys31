FROM python:3.10-slim AS builder
WORKDIR /
COPY ./product/web_backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-slim
WORKDIR /
COPY --from=builder /root/.local /root/.local
COPY ./product/web_backend .
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT [ "python3", "main.py" ]
