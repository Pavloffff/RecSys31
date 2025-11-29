FROM python:3.10-slim AS builder
WORKDIR /
COPY ./product/web/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY ./product/llm_api .

FROM python:3.10-slim
WORKDIR /
COPY --from=builder /root/.local /root/.local
COPY ./product/web .
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
