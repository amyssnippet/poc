import os
import glob
import time
import hashlib
import requests
from typing import Dict, List, Set, Optional, Callable
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

INFERENCE_API_URL = "http://localhost:8000/embed"

MODELS_SYNC_CONFIG = [
    {"key": "clip-base", "collection": "inv_clip_base", "dim": 512, "label": "CLIP ViT-B/32"},
    {"key": "dinov2-base", "collection": "inv_dinov2_base", "dim": 768, "label": "DINOv2 Base"},
    {"key": "clip-large", "collection": "inv_clip_large", "dim": 768, "label": "CLIP ViT-L/14"},
    {"key": "dinov2-large", "collection": "inv_dinov2_large", "dim": 1024, "label": "DINOv2 Large"},
]

def normalize_rel_path(path: str) -> str:
    """Normalizes any path representation to standard forward-slash relative path without leading ./"""
    if not path:
        return ""
    p = os.path.normpath(path).replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p

def scan_disk_catalog(data_dir: str = "./Jewellery_Data") -> List[Dict]:
    """Scans Jewellery_Data directory for all valid jewellery images."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.AVIF")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    catalog = []
    seen = set()
    for path in sorted(files):
        norm_p = normalize_rel_path(path)
        if norm_p in seen:
            continue
        seen.add(norm_p)
        abs_p = os.path.abspath(path)
        sku = os.path.splitext(os.path.basename(path))[0]
        cat = os.path.basename(os.path.dirname(path))
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

def check_catalog_sync_status(client: QdrantClient, data_dir: str = "./Jewellery_Data") -> Dict:
    """Compares files on disk against indexed vectors across all models."""
    disk_items = scan_disk_catalog(data_dir)
    
    status_by_model = {}
    all_unindexed_items = {}
    
    for m in MODELS_SYNC_CONFIG:
        col = m["collection"]
        indexed = get_indexed_paths(client, col)
        
        unindexed = []
        for item in disk_items:
            # Check normalized path, absolute path, or category/sku tuple
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

def fetch_embedding(img_path: str, model_name: str, remove_bg: bool = False, max_retries: int = 3) -> Optional[List[float]]:
    """Calls Inference API to extract embedding vector."""
    for attempt in range(max_retries):
        try:
            with open(img_path, "rb") as f:
                r = requests.post(
                    INFERENCE_API_URL,
                    params={"model_name": model_name, "remove_bg": str(remove_bg).lower()},
                    files={"file": f},
                    timeout=60
                )
                r.raise_for_status()
                return r.json()["vector"]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed embedding {img_path} with {model_name}: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None

def sync_new_items_to_qdrant(
    client: QdrantClient,
    items_to_sync: List[Dict],
    models: Optional[List[Dict]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Incrementally indexes new items across models into Qdrant.
    """
    if models is None:
        models = MODELS_SYNC_CONFIG
        
    total_steps = len(items_to_sync) * len(models)
    current_step = 0
    errors = []
    
    for m in models:
        m_key = m["key"]
        col = m["collection"]
        points_to_upsert = []
        
        for item in items_to_sync:
            current_step += 1
            path = item["path"]
            sku = item["sku"]
            cat = item["category"]
            
            if progress_callback:
                progress_callback(current_step, total_steps, f"Indexing {sku} ({m['label']})...")
                
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
                            "synced_at": time.time()
                        }
                    ))
            except Exception as e:
                errors.append(f"{sku} ({m_key}): {str(e)}")
                
        if points_to_upsert:
            client.upsert(
                collection_name=col,
                points=points_to_upsert
            )
            
    return {
        "synced_count": len(items_to_sync),
        "total_upserts": total_steps - len(errors),
        "errors": errors,
        "items": items_to_sync
    }

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="Jewellery Vector Database Manual Sync Tool")
    parser.add_argument("--check", action="store_true", help="Check sync status without indexing")
    parser.add_argument("--force", action="store_true", help="Force re-sync of all catalog items")
    parser.add_argument("--host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port (default: 6333)")
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port, timeout=60)
    print("\n==================================================")
    print("💎 AI Jewellery Catalog Sync Manager (Terminal CLI)")
    print("==================================================")
    
    status = check_catalog_sync_status(client)
    print(f"📁 Total images in ./Jewellery_Data: {status['total_disk_items']}")
    
    for m_key, m_stat in status["models_status"].items():
        print(f"  • {m_stat['label']:<16} ({m_stat['collection']}): {m_stat['total_indexed']}/{m_stat['total_disk']} indexed")
        
    if args.check:
        print("\nStatus check complete.")
        exit(0)
        
    items_to_sync = status["disk_items"] if args.force else status["new_items"]
    
    if not items_to_sync:
        print("\n✅ All jewellery images are already 100% indexed across all 4 vector models!")
        print("💡 Drop any new .jpg/.png files into ./Jewellery_Data/ and re-run this command to sync them.")
    else:
        print(f"\n⚡ Syncing {len(items_to_sync)} item(s) across all 4 models...")
        pbar = tqdm(total=len(items_to_sync) * len(MODELS_SYNC_CONFIG), desc="Syncing")
        def on_prog(cur, tot, msg):
            pbar.set_description(msg[:40])
            pbar.update(1)
            
        res = sync_new_items_to_qdrant(client, items_to_sync, progress_callback=on_prog)
        pbar.close()
        print(f"\n🎉 Successfully synced {res['synced_count']} item(s) into Qdrant collections!")
        if res["errors"]:
            print(f"⚠️ Errors encountered: {res['errors']}")

