from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

app = FastAPI(title="Inference API")

# Load the model into memory exactly once when the container starts
print("Loading CLIP model into memory...")
model = SentenceTransformer('clip-ViT-B-32')
print("Model loaded and ready.")

@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    # Read the incoming image bytes
    contents = await file.read()
    
    # Convert bytes to a standard RGB image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Generate the vector embedding (a 512-dimensional array for this CLIP model)
    vector = model.encode(image).tolist()
    
    return {"vector": vector}
