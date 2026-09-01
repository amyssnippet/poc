import sys
import os
import requests
from qdrant_client import QdrantClient

# Configuration Constants
INFERENCE_API_URL = "http://localhost:8000/embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "inventory"
TOP_K = 5

def get_image_embedding(image_path: str) -> list[float]:
    """
    Sends a query image to the local Inference API endpoint to obtain its vector embedding.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found at path: {image_path}")

    with open(image_path, "rb") as f:
        response = requests.post(INFERENCE_API_URL, files={"file": f})
        response.raise_for_status()
        return response.json()["vector"]

def search_similar_images(query_vector: list[float], top_k: int = TOP_K):
    """
    Queries Qdrant vector database for nearest neighbors to the query vector.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Execute vector search in Qdrant
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        return response.points
    else:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

def find_default_sample() -> str:
    """
    Finds a default query image if none is passed via CLI.
    """
    candidates = [
        "./Jewellery_Data/ring/ring_001.jpg",
        "./Jewellery_Data/ring/ring_081.jpg",
        "./Jewellery_Data/necklace/necklace_001.jpg",
        "./test/shark.jpg",
        "./poc_inventory/NAT_00001.jpg"
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def main():
    # Parse image path from command line arguments or default to a sample image
    if len(sys.argv) > 1:
        query_image_path = sys.argv[1]
    else:
        query_image_path = find_default_sample()
        print(f"No image path provided. Defaulting to sample image: {query_image_path}")

    print(f"\nProcessing query image: '{query_image_path}'...")
    try:
        query_vector = get_image_embedding(query_image_path)
    except Exception as e:
        print(f"Error generating embedding from Inference API: {e}")
        sys.exit(1)

    print(f"Querying Qdrant collection '{COLLECTION_NAME}' for Top {TOP_K} matches...\n")
    results = search_similar_images(query_vector, top_k=TOP_K)

    print("=" * 80)
    print(f"{'RANK':<5} | {'SKU':<18} | {'CATEGORY':<12} | {'SIMILARITY SCORE':<16} | {'PATH':<20}")
    print("=" * 80)

    for rank, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        sku = payload.get("sku", f"ID_{hit.id}")
        category = payload.get("category", "N/A")
        path = payload.get("path", "N/A")
        score = hit.score
        print(f"{rank:<5} | {sku:<18} | {category:<12} | {score:<16.6f} | {path:<20}")

    print("=" * 80)

if __name__ == "__main__":
    main()
