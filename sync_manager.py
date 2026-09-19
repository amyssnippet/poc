import os
import glob
import time
import hashlib
import requests
from typing import Dict, List, Set, Optional, Callable
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/embed")

# Target DINOv2 Large (1024 dimensions)
MODELS_SYNC_CONFIG = [
    {
        "key": "dinov2-large",
        "collection": "inv_dinov2_large",
        "dim": 1024,
        "label": "Meta DINOv2 Large (1024d)"
    }
]

DEFAULT_DATA_DIR = "./new_data"

def normalize_rel_path(path: str) -> str:
    """Normalizes any path representation to standard forward-slash relative path without leading ./"""
    if not path:
        return ""
    p = os.path.normpath(path).replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p

def extract_category(path: str, sku: str) -> str:
    """Extracts jewellery category from folder structure or SKU prefix."""
    parent = os.path.basename(os.path.dirname(path))
    if parent.lower() not in ("image", "images", "new_data", "data", ".", ""):
        return parent.lower()
    
    # Auto-classify based on SKU prefix from new_data
    s = sku.upper()
    if any(s.startswith(p) for p in ["CRN", "DRN", "DFR", "DHR", "MRN", "R00", "MPR"]):
        return "ring"
    elif any(s.startswith(p) for p in ["CNK", "DNK", "DFN", "NFD"]):
        return "necklace"
    elif any(s.startswith(p) for p in ["CER", "DER", "DFE", "DHE"]):
        return "earring"
    elif any(s.startswith(p) for p in ["CBG", "DBG", "MBG", "DFB"]):
        return "bangle"
    elif any(s.startswith(p) for p in ["DBR", "MBR"]):
        return "bracelet"
    elif any(s.startswith(p) for p in ["CPE", "DPE"]):
        return "pendant"
    elif any(s.startswith(p) for p in ["CTN", "DTN"]):
        return "tanmaniya"
    return "jewellery"

def scan_disk_catalog(data_dir: str = DEFAULT_DATA_DIR) -> List[Dict]:
    """Scans dataset directory for all valid jewellery images."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.AVIF")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    catalog = []
    seen = set()
    for path in sorted(files):
        # Ignore temporary or metadata files
        if "Thumbs.db" in path or os.path.basename(path).startswith("."):
            continue
        norm_p = normalize_rel_path(path)
        if norm_p in seen:
            continue
        seen.add(norm_p)
        abs_p = os.path.abspath(path)
        sku = os.path.splitext(os.path.basename(path))[0]
        cat = extract_category(path, sku)
        mtime = os.path.getmtime(path)
        catalog.append({
            "path": norm_p,
            "abs_path": abs_p,
            "sku": sku,
            "category": cat,
            "mtime": mtime
        })
    return catalog

def get_indexed_paths(client: QdrantClient, collection_name: str) -> Set[str]:
    """Retrieves all indexed image identifiers in a given Qdrant collection."""
    indexed_keys = set()
    if not client.collection_exists(collection_name):
        return indexed_keys
    
    offset = None
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=["path", "sku", "category"],
                with_vectors=False,
                offset=offset
            )
            for r in records:
                payload = r.payload or {}
                p = payload.get("path")
                sku = payload.get("sku")
                cat = payload.get("category")
                if p:
                    indexed_keys.add(normalize_rel_path(p))
                    indexed_keys.add(os.path.abspath(p))
                if sku and cat:
                    indexed_keys.add(f"{cat}/{sku}")
                    indexed_keys.add(sku)
            if next_offset is None:
                break
            offset = next_offset
        except Exception as e:
            print(f"Error scrolling {collection_name}: {e}")
            break
    return indexed_keys

def check_catalog_sync_status(client: QdrantClient, data_dir: str = DEFAULT_DATA_DIR) -> Dict:
    """Compares files on disk against indexed vectors in Qdrant for DINOv2 Large."""
    disk_items = scan_disk_catalog(data_dir)
    
    status_by_model = {}
    all_unindexed_items = {}
    
    for m in MODELS_SYNC_CONFIG:
        col = m["collection"]
        # Ensure collection exists
        if not client.collection_exists(col):
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=m["dim"], distance=Distance.COSINE)
            )
            
        indexed = get_indexed_paths(client, col)
        
        unindexed = []
        for item in disk_items:
            p_match = item["path"] in indexed or item["abs_path"] in indexed
            cat_sku_match = f"{item['category']}/{item['sku']}" in indexed or item["sku"] in indexed
            
            if not p_match and not cat_sku_match:
                unindexed.append(item)
                all_unindexed_items[item["path"]] = item
                
        status_by_model[m["key"]] = {
            "label": m["label"],
            "collection": col,
            "total_disk": len(disk_items),
            "total_indexed": len(disk_items) - len(unindexed),
            "unindexed_count": len(unindexed),
            "unindexed_items": unindexed
        }
        
    return {
        "total_disk_items": len(disk_items),
        "disk_items": disk_items,
        "new_items": list(all_unindexed_items.values()),
        "has_unindexed": len(all_unindexed_items) > 0,
        "models_status": status_by_model
    }

def generate_point_id(rel_path: str) -> int:
    """Generates a deterministic 63-bit integer Point ID from file path."""
    return int(hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:14], 16)

def fetch_embedding(img_path: str, model_name: str = "dinov2-large", remove_bg: bool = False, max_retries: int = 3) -> Optional[List[float]]:
    """Calls Inference API to extract DINOv2 Large 1024-dim embedding vector."""
    for attempt in range(max_retries):
        try:
            with open(img_path, "rb") as f:
                r = requests.post(
                    INFERENCE_API_URL,
                    params={"model_name": model_name, "remove_bg": str(remove_bg).lower()},
                    files={"file": f},
                    timeout=120
                )
                r.raise_for_status()
                return r.json()["vector"]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed embedding {img_path} with {model_name}: {e}")
                return None
            time.sleep(2.0 * (attempt + 1))
    return None

def sync_new_items_to_qdrant(
    client: QdrantClient,
    items_to_sync: List[Dict],
    models: Optional[List[Dict]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Incrementally indexes items into Qdrant using DINOv2 Large (1024d).
    """
    if models is None:
        models = MODELS_SYNC_CONFIG
        
    total_steps = len(items_to_sync) * len(models)
    current_step = 0
    errors = []
    
    for m in models:
        m_key = m["key"]
        col = m["collection"]
        
        # Ensure collection exists with 1024 dimensions
        if not client.collection_exists(col):
            client.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=m["dim"], distance=Distance.COSINE)
            )
            
        points_to_upsert = []
        
        for item in items_to_sync:
            current_step += 1
            path = item["path"]
            sku = item["sku"]
            cat = item["category"]
            
            if progress_callback:
                progress_callback(current_step, total_steps, f"Embedding {sku} with {m['label']}...")
                
            try:
                vec = fetch_embedding(path, m_key)
                if vec is not None:
                    p_id = generate_point_id(path)
                    points_to_upsert.append(PointStruct(
                        id=p_id,
                        vector=vec,
                        payload={
                            "sku": sku,
                            "category": cat,
                            "path": path,
                            "synced_at": time.time(),
                            "model": m_key
                        }
                    ))
                    
                # Upsert in chunks of 25 to avoid large request payloads
                if len(points_to_upsert) >= 25:
                    client.upsert(collection_name=col, points=points_to_upsert)
                    points_to_upsert = []
            except Exception as e:
                errors.append(f"{sku} ({m_key}): {str(e)}")
                
        if points_to_upsert:
            client.upsert(collection_name=col, points=points_to_upsert)
            
    return {
        "synced_count": len(items_to_sync),
        "total_upserts": total_steps - len(errors),
        "errors": errors,
        "items": items_to_sync
    }

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="DINOv2 Large (1024d) Jewellery Catalog Vector Sync Tool")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help=f"Directory to scan (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--check", action="store_true", help="Check sync status without indexing")
    parser.add_argument("--force", action="store_true", help="Force re-sync of all catalog items")
    parser.add_argument("--host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port (default: 6333)")
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port, timeout=60)
    print("\n==================================================================")
    print("💎 DINOv2 Large (1024d) Jewellery Catalog Sync Manager")
    print("==================================================================")
    
    status = check_catalog_sync_status(client, data_dir=args.data_dir)
    print(f"📁 Target dataset: {args.data_dir}")
    print(f"📁 Total images detected on disk: {status['total_disk_items']}")
    
    for m_key, m_stat in status["models_status"].items():
        print(f"  • {m_stat['label']} ({m_stat['collection']}): {m_stat['total_indexed']}/{m_stat['total_disk']} indexed")
        
    if args.check:
        print("\nStatus check complete.")
        exit(0)
        
    items_to_sync = status["disk_items"] if args.force else status["new_items"]
    
    if not items_to_sync:
        print("\n✅ All jewellery images are already 100% indexed in Qdrant with DINOv2 Large (1024d)!")
        print("💡 Drop any new .jpg/.png files into new_data/ and re-run this command to sync them.")
    else:
        print(f"\n⚡ Syncing {len(items_to_sync)} item(s) using DINOv2 Large (1024d)...")
        pbar = tqdm(total=len(items_to_sync), desc="DINOv2 Embedding")
        def on_prog(cur, tot, msg):
            pbar.set_description(msg[:45])
            pbar.update(1)
            
        res = sync_new_items_to_qdrant(client, items_to_sync, progress_callback=on_prog)
        pbar.close()
        print(f"\n🎉 Successfully synced {res['synced_count']} item(s) into Qdrant collection 'inv_dinov2_large'!")
        if res["errors"]:
            print(f"⚠️ Errors encountered: {res['errors']}")
