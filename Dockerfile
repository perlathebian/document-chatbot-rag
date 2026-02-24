# Base image — Python 3.11 slim keeps the container small
FROM python:3.11-slim

# Setting working directory inside the container
WORKDIR /app

# Copying requirements first (Docker caches this layer)
# If requirements.txt hasnt changed, it wont reinstall packages on rebuild
COPY requirements.txt .

# Installing dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copying rest of project
COPY . .

# Creating necessary directories that are gitignored
RUN mkdir -p db/vectordb data/uploads data/sample_pdfs

# Exposing FastAPI port
EXPOSE 8000

# Starting the FastAPI server
# host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]