import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from sync_manager import scan_disk_catalog, generate_point_id, extract_category

# Configuration Constants
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/embed")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "inv_dinov2_large"
MODEL_NAME = "dinov2-large"
VECTOR_SIZE = 1024
BATCH_SIZE = 25
MAX_WORKERS = 4

def get_image_embedding(image_path: str, max_retries: int = 4) -> list[float]:
    """
    Sends an image file to the Inference API to extract its 1024-dim DINOv2 Large vector.
    """
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    INFERENCE_API_URL,
                    params={"model_name": MODEL_NAME, "remove_bg": "false"},
                    files={"file": f},
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["vector"]
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.5 * (attempt + 1))

def process_item(item: dict) -> PointStruct:
    """
    Embeds image and returns a Qdrant PointStruct with 63-bit deterministic ID and rich metadata.
    """
    vector = get_image_embedding(item["abs_path"])
    p_id = generate_point_id(item["path"])
    return PointStruct(
        id=p_id,
        vector=vector,
        payload={
            "sku": item["sku"],
            "category": item["category"],
            "path": item["path"],
            "model": MODEL_NAME,
            "synced_at": time.time()
        }
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest images into Qdrant using DINOv2 Large (1024d)")
    parser.add_argument("target_dir", nargs="?", default="./new_data", help="Target dataset directory (default: ./new_data)")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection to remove old data")
    args = parser.parse_args()

    target_dir = args.target_dir
    print(f"\n==================================================================")
    print(f"💎 DINOv2 Large (1024-dim) Catalog Ingestion Engine")
    print(f"📁 Target Dataset: '{target_dir}'")
    print(f"📦 Vector Collection: '{COLLECTION_NAME}' (Size: {VECTOR_SIZE}, Metric: Cosine)")
    print(f"==================================================================")

    items = scan_disk_catalog(target_dir)
    total_images = len(items)

    if total_images == 0:
        print(f"❌ No valid images found in '{target_dir}'. Exiting.")
        return

    print(f"Discovered {total_images} jewellery images in '{target_dir}'.")
    categories = {}
    for it in items:
        cat = it["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("Catalog category breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat.capitalize()}: {count} items")

    print("\nConnecting to Qdrant Vector Database...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    if args.recreate and client.collection_exists(COLLECTION_NAME):
        print(f"Recreating collection '{COLLECTION_NAME}' as requested...")
        client.delete_collection(COLLECTION_NAME)

    # Ensure the collection exists with 1024 dimensions
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating new collection '{COLLECTION_NAME}' (dim={VECTOR_SIZE}, metric=Cosine)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    else:
        print(f"Using existing collection '{COLLECTION_NAME}'.")

    print(f"\nBeginning ingestion using DINOv2 Large ({MAX_WORKERS} workers, batch size {BATCH_SIZE})...")
    points_to_upsert = []
    
    with tqdm(total=total_images, desc="Ingesting DINOv2 Vectors", unit="img") as pbar:
        for i in range(0, total_images, BATCH_SIZE):
            chunk = items[i:i + BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_item, it) for it in chunk]
                for future in futures:
                    try:
                        pt = future.result()
                        points_to_upsert.append(pt)
                    except Exception as e:
                        print(f"\n⚠️ Error processing item: {e}")
                    pbar.update(1)

            if points_to_upsert:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_to_upsert
                )
                points_to_upsert = []

    final_count = client.count(COLLECTION_NAME).count
    print(f"\n🎉 Ingestion complete! Qdrant collection '{COLLECTION_NAME}' now holds {final_count} vectors.")

if __name__ == "__main__":
    main()
