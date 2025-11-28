FROM python:3.10-slim AS builder
WORKDIR /
COPY ./product/llm_api/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY ./product/llm_api .

FROM python:3.10-slim
WORKDIR /
COPY --from=builder /root/.local /root/.local
COPY ./product/llm_api .
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT [ "python3", "main.py" ]