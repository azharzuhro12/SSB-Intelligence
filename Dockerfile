FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgomp1 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python run_all.py
CMD ["uvicorn", "ssb.api:app", "--host", "0.0.0.0", "--port", "8000"]
