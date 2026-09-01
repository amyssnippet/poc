import os
import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

# Configuration Constants
INFERENCE_API_URL = "http://localhost:8000/embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "inventory"
VECTOR_SIZE = 512
BATCH_SIZE = 100
MAX_WORKERS = 8

def get_image_embedding(image_path: str, max_retries: int = 5) -> list[float]:
    """
    Sends an image file to the local Inference API endpoint to obtain its CLIP vector with retry handling.
    """
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                response = requests.post(INFERENCE_API_URL, files={"file": f}, timeout=30)
                response.raise_for_status()
                return response.json()["vector"]
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise e
            import time
            time.sleep(1.0 * (attempt + 1))


def find_all_images(base_dir: str) -> list[dict]:
    """
    Recursively scans the target directory for image files.
    Returns a list of dicts with file path metadata.
    """
    image_items = []
    supported_exts = (".jpg", ".jpeg", ".png", ".webp")

    for root, _, files in os.walk(base_dir):
        for file in sorted(files):
            if file.lower().endswith(supported_exts):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=base_dir)
                sku_name = os.path.splitext(file)[0]
                
                # Category is parent subfolder name if nested, otherwise root directory
                rel_dir = os.path.dirname(rel_path)
                category = rel_dir if rel_dir else "default"

                image_items.append({
                    "full_path": full_path,
                    "rel_path": full_path,
                    "filename": file,
                    "sku": sku_name,
                    "category": category
                })

    return sorted(image_items, key=lambda x: x["full_path"])

def process_image_item(idx: int, item: dict) -> PointStruct:
    """
    Processes a single image item: fetches embedding and constructs a Qdrant PointStruct.
    """
    vector = get_image_embedding(item["full_path"])

    return PointStruct(
        id=idx,
        vector=vector,
        payload={
            "sku": item["sku"],
            "category": item["category"],
            "path": item["rel_path"]
        }
    )

def main():
    # Determine target directory from CLI argument or default locations
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    elif os.path.exists("./Jewellery_Data"):
        target_dir = "./Jewellery_Data"
    else:
        target_dir = "./poc_inventory"

    print(f"Target dataset directory: '{target_dir}'")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Directory '{target_dir}' does not exist.")

    image_items = find_all_images(target_dir)
    total_images = len(image_items)

    if total_images == 0:
        print(f"No image files found in '{target_dir}'. Exiting.")
        return

    print(f"Found {total_images} images across subdirectories for ingestion.")

    print("\nConnecting to Qdrant Vector Database...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate the 'inventory' collection with 512-dim vectors and Cosine distance
    print(f"Creating collection '{COLLECTION_NAME}' (dim={VECTOR_SIZE}, metric=Cosine)...")
    if hasattr(client, "collection_exists") and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        try:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        except Exception:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    print(f"Beginning parallel ingestion into Qdrant ({MAX_WORKERS} workers, batch size {BATCH_SIZE})...")
    with tqdm(total=total_images, desc="Ingesting Jewellery Data", unit="img") as pbar:
        for i in range(0, total_images, BATCH_SIZE):
            chunk = image_items[i:i + BATCH_SIZE]
            points_batch = []

            # Embed batch in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Assign 1-indexed unique integer point IDs (i + idx + 1)
                futures = [
                    executor.submit(process_image_item, i + idx + 1, item)
                    for idx, item in enumerate(chunk)
                ]
                for future in futures:
                    points_batch.append(future.result())
                    pbar.update(1)

            # Upsert current batch to Qdrant
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )

    print(f"\nIngestion complete! Successfully indexed {total_images} vectors into Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
