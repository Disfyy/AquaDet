FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md /app/
COPY backend /app/backend
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
