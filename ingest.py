import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

# Configuration Constants
INVENTORY_DIR = "./poc_inventory"
INFERENCE_API_URL = "http://localhost:8000/embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "inventory"
VECTOR_SIZE = 512
BATCH_SIZE = 100
MAX_WORKERS = 8

def get_image_embedding(image_path: str) -> list[float]:
    """
    Sends an image file to the local Inference API endpoint to obtain its CLIP vector.
    """
    with open(image_path, "rb") as f:
        response = requests.post(INFERENCE_API_URL, files={"file": f})
        response.raise_for_status()
        return response.json()["vector"]

def extract_sku_id(filename: str) -> int:
    """
    Extracts the numeric part of the filename (e.g., 1 from 'NAT_00001.jpg') to use as Qdrant Point ID.
    """
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group(0))
    raise ValueError(f"Could not extract numeric ID from filename: {filename}")

def process_file(filename: str) -> PointStruct:
    """
    Processes a single image: fetches embedding and constructs a Qdrant PointStruct.
    """
    image_path = os.path.join(INVENTORY_DIR, filename)
    sku_name = os.path.splitext(filename)[0]  # e.g., "NAT_00001"
    point_id = extract_sku_id(filename)      # e.g., 1

    vector = get_image_embedding(image_path)

    return PointStruct(
        id=point_id,
        vector=vector,
        payload={"sku": sku_name}
    )


def main():
    print("Connecting to Qdrant Vector Database...")
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

    # Find all image files in ./poc_inventory/
    if not os.path.exists(INVENTORY_DIR):
        raise FileNotFoundError(f"Directory {INVENTORY_DIR} does not exist.")

    image_files = sorted([
        f for f in os.listdir(INVENTORY_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    total_images = len(image_files)
    print(f"Found {total_images} images in '{INVENTORY_DIR}' for ingestion.")

    session = requests.Session()

    print(f"Beginning parallel ingestion into Qdrant ({MAX_WORKERS} workers, batch size {BATCH_SIZE})...")
    with tqdm(total=total_images, desc="Ingesting Inventory", unit="img") as pbar:
        for i in range(0, total_images, BATCH_SIZE):
            chunk = image_files[i:i + BATCH_SIZE]
            points_batch = []

            # Embed batch in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_file, fname) for fname in chunk]
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

