import os
import sys
import glob
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

INFERENCE_API_URL = "http://localhost:8000/embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

MODELS = [
    {"name": "clip-base", "collection": "inv_clip_base", "dim": 512},
    {"name": "dinov2-base", "collection": "inv_dinov2_base", "dim": 768},
    {"name": "clip-large", "collection": "inv_clip_large", "dim": 768},
    {"name": "dinov2-large", "collection": "inv_dinov2_large", "dim": 1024},
]

def get_embedding(img_path: str, model_name: str, max_retries: int = 5) -> list[float]:
    for attempt in range(max_retries):
        try:
            with open(img_path, "rb") as f:
                r = requests.post(
                    INFERENCE_API_URL,
                    params={"model_name": model_name},
                    files={"file": f},
                    timeout=120
                )
                r.raise_for_status()
                return r.json()["vector"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 * (attempt + 1))

def ingest_collection(client: QdrantClient, model_name: str, col_name: str, dim: int, images: list[str]):
    print(f"\n==================================================")
    print(f"Indexing {len(images)} images for {model_name} -> {col_name} (dim: {dim})")
    print(f"==================================================")
    
    if client.collection_exists(col_name):
        client.delete_collection(col_name)
        
    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    def process_item(item_tuple):
        idx, path = item_tuple
        sku = os.path.splitext(os.path.basename(path))[0]
        cat = os.path.basename(os.path.dirname(path))
        vec = get_embedding(path, model_name)
        return PointStruct(
            id=idx + 1,
            vector=vec,
            payload={"sku": sku, "category": cat, "path": path}
        )

    points = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = executor.map(process_item, enumerate(images))
        for point in tqdm(futures, total=len(images), desc=f"Ingesting ({model_name})"):
            points.append(point)

    # Batch upsert into Qdrant
    for i in range(0, len(points), 50):
        client.upsert(collection_name=col_name, points=points[i:i+50])
        
    count = client.count(col_name).count
    print(f"✅ Successfully indexed {count} items into collection '{col_name}'!")

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./Jewellery_Data"
    images = glob.glob(f"{target_dir}/**/*.jpg", recursive=True) + glob.glob(f"{target_dir}/**/*.jpeg", recursive=True)
    images = sorted(list(set(images)))
    print(f"Discovered {len(images)} jewellery images in '{target_dir}'.")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    
    for m in MODELS:
        ingest_collection(client, m["name"], m["collection"], m["dim"], images)

    print("\n🎉 ALL 4 MODEL COLLECTIONS SUCCESSFULLY INGESTED & READY IN QDRANT!")

if __name__ == "__main__":
    main()
