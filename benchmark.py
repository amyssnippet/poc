import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

INFERENCE_API_URL = "http://localhost:8000/embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
BATCH_SIZE = 50
MAX_WORKERS = 6

MODELS = [
    {"name": "clip-base", "collection": "inv_clip_base", "dim": 512, "desc": "CLIP ViT-B/32 (Baseline)"},
    {"name": "clip-large", "collection": "inv_clip_large", "dim": 768, "desc": "CLIP ViT-L/14 (High-Res CLIP)"},
    {"name": "dinov2-base", "collection": "inv_dinov2_base", "dim": 768, "desc": "Meta DINOv2-Base (Self-Supervised)"},
    {"name": "dinov2-large", "collection": "inv_dinov2_large", "dim": 1024, "desc": "Meta DINOv2-Large (High-Res Geometry)"},
]

def get_image_embedding(image_path: str, model_name: str, max_retries: int = 5) -> list[float]:
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    INFERENCE_API_URL,
                    params={"model_name": model_name},
                    files={"file": f},
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["vector"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            import time
            time.sleep(1.5 * (attempt + 1))


def find_all_images(base_dir: str) -> list[dict]:
    image_items = []
    supported_exts = (".jpg", ".jpeg", ".png", ".webp", ".avif")

    for root, _, files in os.walk(base_dir):
        for file in sorted(files):
            if file.lower().endswith(supported_exts):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=base_dir)
                sku_name = os.path.splitext(file)[0]
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

def process_item(idx: int, item: dict, model_name: str) -> PointStruct:
    vector = get_image_embedding(item["full_path"], model_name)
    return PointStruct(
        id=idx,
        vector=vector,
        payload={
            "sku": item["sku"],
            "category": item["category"],
            "path": item["rel_path"]
        }
    )

def ingest_model_dataset(client: QdrantClient, model_info: dict, image_items: list[dict]):
    col_name = model_info["collection"]
    dim = model_info["dim"]
    model_name = model_info["name"]

    print(f"\n--- Indexing for Model: {model_info['desc']} ({model_name}) ---")
    if hasattr(client, "collection_exists") and client.collection_exists(col_name):
        client.delete_collection(col_name)
    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    total = len(image_items)
    with tqdm(total=total, desc=f"Ingesting ({model_name})", unit="img") as pbar:
        for i in range(0, total, BATCH_SIZE):
            chunk = image_items[i:i + BATCH_SIZE]
            points_batch = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(process_item, i + idx + 1, item, model_name)
                    for idx, item in enumerate(chunk)
                ]
                for future in futures:
                    points_batch.append(future.result())
                    pbar.update(1)

            client.upsert(collection_name=col_name, points=points_batch)

def query_model(client: QdrantClient, model_info: dict, query_path: str, top_k: int = 5):
    model_name = model_info["name"]
    col_name = model_info["collection"]

    query_vector = get_image_embedding(query_path, model_name)
    if hasattr(client, "query_points"):
        res = client.query_points(collection_name=col_name, query=query_vector, limit=top_k).points
    else:
        res = client.search(collection_name=col_name, query_vector=query_vector, limit=top_k)

    return res

def run_benchmark(target_dir: str = "./Jewellery_Data", query_images: list[str] = None, max_images: int = 150):
    if query_images is None:
        query_images = [
            "./test/shopping.jpeg",
            "./test/gold-platinum-ring-377062544-zkx5j.jpg.avif",
            "./Jewellery_Data/ring/ring_081.jpg",
        ]

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    all_image_items = find_all_images(target_dir)
    image_items = all_image_items[:max_images] if max_images else all_image_items

    print(f"Loaded {len(image_items)} (out of {len(all_image_items)}) images from '{target_dir}'. Starting Benchmark for {len(MODELS)} models...")

    # Step 1: Ingest dataset for each model
    for model_info in MODELS:
        ingest_model_dataset(client, model_info, image_items)

    # Step 2: Run queries across all models and compare
    for q_path in query_images:
        if not os.path.exists(q_path):
            print(f"Skipping query image '{q_path}' (file not found).")
            continue

        print("\n" + "=" * 95)
        print(f" BENCHMARK COMPARISON FOR QUERY IMAGE: {q_path}")
        print("=" * 95)

        for model_info in MODELS:
            hits = query_model(client, model_info, q_path, top_k=5)
            print(f"\n>> Model: {model_info['desc']} [{model_info['name']}]")
            print(f"{'RANK':<5} | {'SKU':<18} | {'CATEGORY':<12} | {'COSINE SCORE':<14} | {'PATH':<25}")
            print("-" * 80)
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}
                sku = payload.get("sku", f"ID_{hit.id}")
                cat = payload.get("category", "N/A")
                path = payload.get("path", "N/A")
                score = hit.score
                print(f"{rank:<5} | {sku:<18} | {cat:<12} | {score:<14.6f} | {path:<25}")

if __name__ == "__main__":
    dir_arg = sys.argv[1] if len(sys.argv) > 1 else "./Jewellery_Data"
    run_benchmark(dir_arg, max_images=None)


