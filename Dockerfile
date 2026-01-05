# ---------------------------
# Dockerfile
# ---------------------------

# 1️⃣ Base image
FROM python:3.11-slim

# 2️⃣ Set work directory
WORKDIR /app

# 3️⃣ Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy project files
COPY . .

# 5️⃣ Expose port
EXPOSE 8000

# 6️⃣ Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# 7️⃣ Start the API server
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
