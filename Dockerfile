FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit
COPY reports ./reports
RUN pip install --no-cache-dir . && useradd --create-home appuser && mkdir /app/.labelops && chown -R appuser /app
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
