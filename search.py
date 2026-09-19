import sys
import os
import requests
from qdrant_client import QdrantClient

# Configuration Constants
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/embed")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "inv_dinov2_large"
MODEL_NAME = "dinov2-large"
TOP_K = 5

def get_image_embedding(image_path: str, remove_bg: bool = False) -> list[float]:
    """
    Sends a query image to the local Inference API endpoint to obtain its DINOv2 Large 1024d embedding.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found at path: {image_path}")

    with open(image_path, "rb") as f:
        response = requests.post(
            INFERENCE_API_URL,
            params={"model_name": MODEL_NAME, "remove_bg": str(remove_bg).lower()},
            files={"file": f},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["vector"]

def search_similar_images(query_vector: list[float], top_k: int = TOP_K):
    """
    Queries Qdrant vector database for nearest neighbors to the query vector in inv_dinov2_large.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return response.points
    else:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

def find_default_sample() -> str:
    """
    Finds a default query image if none is passed via CLI.
    """
    candidates = [
        "./test/shopping.jpeg",
        "./test/gold-platinum-ring-377062544-zkx5j.jpg.avif",
        "./test/images.jpeg",
        "./new_data/image/CRN00143.jpg",
        "./new_data/image/DBG00015.jpg",
        "./Jewellery_Data/ring/ring_081.jpg"
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def main():
    if len(sys.argv) > 1:
        query_image_path = sys.argv[1]
    else:
        query_image_path = find_default_sample()
        print(f"No image path provided. Defaulting to sample image: {query_image_path}")

    print(f"\nProcessing query image: '{query_image_path}' using DINOv2 Large (1024d)...")
    try:
        query_vector = get_image_embedding(query_image_path)
    except Exception as e:
        print(f"Error generating embedding from Inference API: {e}")
        sys.exit(1)

    print(f"Querying Qdrant collection '{COLLECTION_NAME}' for Top {TOP_K} matches...\n")
    results = search_similar_images(query_vector, top_k=TOP_K)

    print("=" * 85)
    print(f"{'RANK':<5} | {'SKU':<20} | {'CATEGORY':<12} | {'SIMILARITY':<14} | {'PATH':<25}")
    print("=" * 85)

    for rank, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        sku = payload.get("sku", f"ID_{hit.id}")
        category = payload.get("category", "N/A")
        path = payload.get("path", "N/A")
        score = hit.score
        score_pct = f"{score * 100:.2f}%"
        print(f"{rank:<5} | {sku:<20} | {category:<12} | {score_pct:<14} | {path:<25}")

    print("=" * 85)

if __name__ == "__main__":
    main()
