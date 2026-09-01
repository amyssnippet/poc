import io
import base64
from fastapi import FastAPI, UploadFile, File, Query
from PIL import Image

app = FastAPI(title="Multi-Model Inference API")

# Global Cache for Models
models_cache = {}
processors_cache = {}
_rembg_func = None

@app.get("/health")
def health():
    return {"status": "ok", "cached_models": list(models_cache.keys())}

def get_rembg():
    global _rembg_func
    if _rembg_func is None:
        try:
            from rembg import remove
            _rembg_func = remove
            print("rembg loaded successfully.")
        except Exception as e:
            print(f"rembg import error: {e}")
            _rembg_func = False
    return _rembg_func

def get_model(model_name: str):
    if model_name in models_cache:
        return models_cache[model_name]
    
    print(f"Loading model '{model_name}' into memory...")
    if model_name == "clip-base":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('clip-ViT-B-32')
        models_cache[model_name] = model
    elif model_name == "clip-large":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('clip-ViT-L-14')
        models_cache[model_name] = model
    elif model_name == "dinov2-base":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        processors_cache[model_name] = processor
        models_cache[model_name] = model
    elif model_name == "dinov2-large":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        model = AutoModel.from_pretrained('facebook/dinov2-large')
        model.eval()
        processors_cache[model_name] = processor
        models_cache[model_name] = model

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print(f"Model '{model_name}' loaded successfully.")
    return models_cache[model_name]

def process_background_removal(image: Image.Image) -> Image.Image:
    rembg_fn = get_rembg()
    if not rembg_fn:
        return image.convert("RGB")
    try:
        # Resize to max 512x512 for sub-second CPU background removal
        img_copy = image.copy()
        img_copy.thumbnail((512, 512), Image.Resampling.LANCZOS)
        rgba = rembg_fn(img_copy)
        
        # 1. Detect non-transparent bounding box to Auto-Focus & Zoom into the jewellery
        alpha = rgba.split()[3] if rgba.mode == "RGBA" else None
        if alpha:
            bbox = alpha.getbbox()
            if bbox:
                # Add small 5% margin around the piece
                pw = max(4, int((bbox[2] - bbox[0]) * 0.05))
                ph = max(4, int((bbox[3] - bbox[1]) * 0.05))
                crop_box = (
                    max(0, bbox[0] - pw),
                    max(0, bbox[1] - ph),
                    min(rgba.width, bbox[2] + pw),
                    min(rgba.height, bbox[3] + ph)
                )
                rgba = rgba.crop(crop_box)

        # 2. Composite RGBA onto pure square white canvas with centered padding
        max_dim = max(rgba.size)
        background = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        offset = ((max_dim - rgba.size[0]) // 2, (max_dim - rgba.size[1]) // 2)
        if rgba.mode == "RGBA":
            background.paste(rgba, offset, mask=rgba.split()[3])
        else:
            background.paste(rgba, offset)
            
        # Resize to standard 512x512 high resolution
        return background.resize((512, 512), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Background removal warning: {e}")
        return image.convert("RGB")



@app.post("/embed")
def embed_image(
    file: UploadFile = File(...),
    model_name: str = Query("clip-base", description="Model variant: clip-base, clip-large, dinov2-base, dinov2-large"),
    remove_bg: bool = Query(False, description="Remove background noise before feature extraction")
):
    contents = file.file.read()
    raw_image = Image.open(io.BytesIO(contents)).convert("RGB")


    if remove_bg:
        image = process_background_removal(raw_image)
    else:
        image = raw_image

    model = get_model(model_name)

    if model_name in ("clip-base", "clip-large"):
        vector = model.encode(image).tolist()
    elif model_name in ("dinov2-base", "dinov2-large"):
        import torch
        processor = processors_cache[model_name]
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            norm_vector = torch.nn.functional.normalize(cls_token, p=2, dim=1)
            vector = norm_vector.squeeze(0).tolist()

    # Convert processed image to base64 for preview
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "model": model_name,
        "vector": vector,
        "dim": len(vector),
        "bg_removed": remove_bg,
        "processed_image_b64": img_b64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

