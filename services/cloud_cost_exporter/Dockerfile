FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY services ./services
RUN pip install -e "." prometheus_client "uvicorn[standard]"
EXPOSE 9109
CMD ["python", "-m", "services.cloud_cost_exporter.app"]
