FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies required for image processing
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the DINOv2-Large model into the image during the build phase
RUN python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-large'); AutoModel.from_pretrained('facebook/dinov2-large')"

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


