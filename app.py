import streamlit as st
import requests
import io
import os
import time
import base64
from PIL import Image
from qdrant_client import QdrantClient
from sync_manager import (
    check_catalog_sync_status,
    sync_new_items_to_qdrant,
    scan_disk_catalog,
    DEFAULT_DATA_DIR,
    extract_category
)

# Page configuration
st.set_page_config(
    page_title="DINOv2 (1024d) AI Jewellery Search Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Luxury Gold & Dark Glassmorphic Dashboard
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #d4af37 0%, #f3e5ab 50%, #aa7c11 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    
    .model-badge-top {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(212, 175, 55, 0.8);
    }
    
    .score-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .score-badge-high {
        background: linear-gradient(135deg, #d4af37 0%, #b8860b 100%);
        color: black;
        font-weight: 700;
    }
    
    .cat-badge {
        background: #334155;
        color: #cbd5e1;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        text-transform: capitalize;
    }
</style>
""", unsafe_allow_html=True)

# Configuration Constants
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/embed")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
MODEL_NAME = "dinov2-large"
COLLECTION_NAME = "inv_dinov2_large"
DATASET_PATH = "./new_data"

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

def embed_image_api(image_bytes: bytes, remove_bg: bool) -> dict:
    files = {"file": ("query.jpg", image_bytes, "image/jpeg")}
    params = {"model_name": MODEL_NAME, "remove_bg": str(remove_bg).lower()}
    response = requests.post(INFERENCE_API_URL, params=params, files=files, timeout=120)
    response.raise_for_status()
    return response.json()

def search_qdrant(client: QdrantClient, vector: list[float], top_k: int = 6):
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return []

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True
            )
            return response.points
        else:
            return client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )
    except Exception as e:
        st.error(f"Qdrant Search Error: {e}")
        return []

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/diamond.png", width=64)
st.sidebar.title("Search Controls")

st.sidebar.markdown(
    """
    <div style="background: rgba(30,41,59,0.8); padding: 10px; border-radius: 8px; border: 1px solid rgba(59,130,246,0.3); margin-bottom: 12px;">
        <span style="font-size: 0.85rem; color: #94a3b8;">Active Vision Model</span><br/>
        <strong style="color: #60a5fa; font-size: 1.05rem;">Meta DINOv2 Large</strong><br/>
        <small style="color: #cbd5e1;">1024-dim • Micro-Texture Specialist</small>
    </div>
    """,
    unsafe_allow_html=True
)

remove_bg_toggle = st.sidebar.checkbox(
    "🧹 Remove Background / Auto-Zoom",
    value=False,
    help="Applies AI Background Isolation (rembg) to isolate the jewellery piece and filter out background reflections, skin, and cloth noise."
)

top_k_slider = st.sidebar.slider("Top Results Count", min_value=3, max_value=15, value=6)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Catalog Sync Manager (`new_data`)")
q_client = get_qdrant_client()

# Check dataset vs Qdrant collection status
sync_status = check_catalog_sync_status(q_client, data_dir=DATASET_PATH)
total_disk = sync_status["total_disk_items"]
has_unindexed = sync_status["has_unindexed"]
new_items_list = sync_status["new_items"]

if has_unindexed:
    st.sidebar.warning(f"🔔 **{len(new_items_list)} Unindexed Image(s) Detected!**")
    if st.sidebar.button("⚡ Index Dataset in Qdrant", type="primary", use_container_width=True):
        with st.spinner("Extracting DINOv2 Large (1024d) vectors & indexing..."):
            res = sync_new_items_to_qdrant(q_client, new_items_list)
            st.sidebar.success(f"✅ Synced {res['synced_count']} items!")
            time.sleep(1)
            st.rerun()
else:
    st.sidebar.success(f"✅ All {total_disk} items indexed in Qdrant (`{COLLECTION_NAME}`)")

if st.sidebar.button("🔄 Rescan `new_data` Folder", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### System Health")
try:
    cols = [c.name for c in q_client.get_collections().collections]
    count_1024 = q_client.count(COLLECTION_NAME).count if COLLECTION_NAME in cols else 0
    st.sidebar.success(f"⚡ Qdrant Online ({count_1024} vectors in 1024d collection)")
except Exception:
    st.sidebar.error("❌ Qdrant Database Offline")

try:
    r_health = requests.get(f"{os.path.dirname(INFERENCE_API_URL)}/health", timeout=2)
    if r_health.status_code == 200:
        st.sidebar.success("⚡ Inference API Online (DINOv2 1024d)")
    else:
        st.sidebar.warning("⚠️ Inference API Starting...")
except Exception:
    st.sidebar.error("❌ Inference API Offline")

# Header Section
st.markdown('<div class="main-header">💎 AI Jewellery Visual Search Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="model-badge-top">🧬 Model: Meta DINOv2 Large (1024-dimensional Patch Embeddings)</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Trained on <code>new_data</code> catalog for fine geometric micro-texture & pattern matching (Grid Mesh • Filigree • Pave Diamonds • Hammered Gold)</div>',
    unsafe_allow_html=True
)

# Unindexed Warning Banner
if has_unindexed:
    st.warning(f"🔔 **Unindexed Images**: Found {len(new_items_list)} unindexed jewellery image(s) in `{DATASET_PATH}`. Index them to make them searchable!")
    with st.expander(f"👁️ View Unindexed Images & Run Batch Indexing", expanded=False):
        preview_count = min(len(new_items_list), 6)
        sync_cols = st.columns(preview_count)
        for idx, item in enumerate(new_items_list[:preview_count]):
            with sync_cols[idx]:
                if os.path.exists(item["path"]):
                    st.image(item["path"], use_container_width=True)
                st.caption(f"**{item['sku']}**\n\n`{item['category']}`")
                
        if st.button("🚀 Index All Unindexed Items Now", type="primary", use_container_width=True):
            pbar = st.progress(0.0)
            status_placeholder = st.empty()
            
            def handle_prog(cur, tot, desc):
                pbar.progress(cur / tot)
                status_placeholder.text(f"[{cur}/{tot}] {desc}")
                
            res = sync_new_items_to_qdrant(q_client, new_items_list, progress_callback=handle_prog)
            st.success(f"🎉 Successfully indexed {res['synced_count']} items with DINOv2 Large!")
            time.sleep(1.5)
            st.rerun()

tab1, tab2, tab3 = st.tabs([
    "🔍 Visual Similarity Search",
    "🔬 Micro-Texture & Detail Inspector",
    "➕ Add & Index New Jewellery"
])

with tab1:
    col_upload, col_preview = st.columns([1, 1])
    
    with col_upload:
        st.markdown("### 1. Query Image")
        uploaded_file = st.file_uploader(
            "Upload Ring, Bangle, Earring, or Necklace Image",
            type=["jpg", "jpeg", "png", "webp", "avif"]
        )

        sample_options = [
            "(None)",
            "Grid Mesh Ring (test/shopping.jpeg)",
            "Gold Platinum Ring (test/gold-platinum-ring.avif)",
            "Model Wearing Ring (test/images.jpeg)",
            "Diamond Ring Sample (new_data/image/CRN00143.jpg)",
            "Diamond Bangle Sample (new_data/image/DBG00015.jpg)",
            "Diamond Earring Sample (new_data/image/DER00268.jpg)"
        ]
        sample_choice = st.selectbox("Or select from test gallery:", sample_options)

    # Determine query image bytes
    query_bytes = None
    query_name = "Uploaded Image"
    
    if uploaded_file is not None:
        query_bytes = uploaded_file.getvalue()
        query_name = uploaded_file.name
    elif sample_choice != "(None)":
        if "shopping.jpeg" in sample_choice:
            img_path = "./test/shopping.jpeg"
        elif "gold-platinum" in sample_choice:
            img_path = "./test/gold-platinum-ring-377062544-zkx5j.jpg.avif"
        elif "images.jpeg" in sample_choice:
            img_path = "./test/images.jpeg"
        elif "CRN00143" in sample_choice:
            img_path = "./new_data/image/CRN00143.jpg"
        elif "DBG00015" in sample_choice:
            img_path = "./new_data/image/DBG00015.jpg"
        elif "DER00268" in sample_choice:
            img_path = "./new_data/image/DER00268.jpg"
        else:
            img_path = "./test/shopping.jpeg"
        
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                query_bytes = f.read()
            query_name = os.path.basename(img_path)

    effective_query_bytes = query_bytes
    if query_bytes:
        orig_img = Image.open(io.BytesIO(query_bytes))
        with col_upload:
            crop_focus_toggle = st.checkbox(
                "🎯 Focus & Zoom on Jewellery (Crop Region of Interest)",
                value=False,
                help="Draw a box directly around the jewellery item on human models to zoom in 100% on the piece!"
            )
            if crop_focus_toggle:
                st.markdown("<small style='color: #a0aec0;'>Drag the box over the ring or necklace to zoom in:</small>", unsafe_allow_html=True)
                try:
                    from streamlit_cropper import st_cropper
                    cropped_pil = st_cropper(orig_img, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
                    if cropped_pil is not None:
                        buf = io.BytesIO()
                        if cropped_pil.mode in ("RGBA", "LA", "P"):
                            rgb_canvas = Image.new("RGB", cropped_pil.size, (255, 255, 255))
                            if cropped_pil.mode == "RGBA":
                                rgb_canvas.paste(cropped_pil, mask=cropped_pil.split()[3])
                            else:
                                rgb_canvas.paste(cropped_pil.convert("RGBA"))
                            rgb_canvas.save(buf, format="JPEG", quality=95)
                        else:
                            cropped_pil.convert("RGB").save(buf, format="JPEG", quality=95)
                        effective_query_bytes = buf.getvalue()
                except Exception as e:
                    st.warning(f"Cropper tool warning: {e}")

    with col_preview:
        if query_bytes:
            st.markdown("### 2. Previews")
            p_col1, p_col2 = st.columns(2)
            show_input = Image.open(io.BytesIO(effective_query_bytes))
            p_col1.image(show_input, caption="Target Query Image", use_container_width=True)
            
            if remove_bg_toggle:
                st.info("🧹 Background Removal & Auto-Centering Enabled")
            else:
                st.warning("Raw Image (Background Removal Disabled)")

    if effective_query_bytes and st.button("🚀 Run DINOv2 Visual Search", type="primary", use_container_width=True):
        with st.spinner("Extracting 1024-dim DINOv2 Large feature embedding..."):
            try:
                res_embed = embed_image_api(effective_query_bytes, remove_bg_toggle)
                vec = res_embed["vector"]
                
                # Show processed background-removed image if available
                if "processed_image_b64" in res_embed and remove_bg_toggle:
                    proc_b64 = res_embed["processed_image_b64"]
                    proc_bytes = base64.b64decode(proc_b64)
                    with col_preview:
                        p_col2.image(Image.open(io.BytesIO(proc_bytes)), caption="AI Cleaned & Centered", use_container_width=True)

                hits = search_qdrant(q_client, vec, top_k=top_k_slider)
                
                st.markdown("---")
                if len(hits) == 0:
                    st.warning(f"No items found in collection '{COLLECTION_NAME}'. Click 'Index Dataset in Qdrant' in the sidebar or run ingest.py.")
                else:
                    st.markdown(f"### Top {len(hits)} Matching Items Retrieved (DINOv2 1024d)")
                    num_cols = min(len(hits), 6)
                    res_cols = st.columns(num_cols)
                    for idx, hit in enumerate(hits):
                        col = res_cols[idx % num_cols]
                        payload = hit.payload or {}
                        sku = payload.get("sku", f"ID_{hit.id}")
                        cat = payload.get("category", "Jewellery")
                        rel_path = payload.get("path", "")
                        score = hit.score
                        score_pct = score * 100
                        
                        with col:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            if rel_path and os.path.exists(rel_path):
                                st.image(rel_path, use_container_width=True)
                            else:
                                st.write("📷 [Image Preview]")
                            
                            st.markdown(f"**{sku}**")
                            st.markdown(f"<span class='cat-badge'>{cat}</span>", unsafe_allow_html=True)
                            st.markdown(f"<div style='margin-top: 6px;'><span class='score-badge score-badge-high'>{score_pct:.2f}% Match</span></div>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Execution Error: {e}")

with tab2:
    st.markdown("### 🔬 Micro-Texture & Detail Inspector")
    st.markdown("Inspect fine geometric patterns (grid mesh, pave diamond layout, prong settings) comparing query against top match.")
    
    if effective_query_bytes:
        if st.button("🔎 Inspect Query vs. Top Match", type="secondary", use_container_width=True):
            try:
                res_embed = embed_image_api(effective_query_bytes, remove_bg_toggle)
                vec = res_embed["vector"]
                hits = search_qdrant(q_client, vec, top_k=1)
                
                if hits:
                    top_hit = hits[0]
                    p = top_hit.payload or {}
                    top_path = p.get("path", "")
                    top_sku = p.get("sku", "")
                    top_score = top_hit.score
                    
                    insp_col1, insp_col2 = st.columns(2)
                    with insp_col1:
                        st.markdown("#### Query Image (Target)")
                        st.image(Image.open(io.BytesIO(effective_query_bytes)), use_container_width=True)
                    with insp_col2:
                        st.markdown(f"#### Top Retrieved Match: **{top_sku}** ({top_score*100:.2f}%)")
                        if top_path and os.path.exists(top_path):
                            st.image(top_path, use_container_width=True)
                        st.info(f"DINOv2 Large matched spatial patch tokens with Cosine similarity score of `{top_score:.4f}`.")
                else:
                    st.warning("No matches in collection. Please index dataset first.")
            except Exception as e:
                st.error(f"Error inspecting: {e}")
    else:
        st.info("Select or upload a query image in Tab 1 to run the Micro-Texture Inspector.")

with tab3:
    st.markdown("### ➕ Add New Jewellery to Catalog (`new_data`)")
    st.markdown("Upload new products directly to `new_data`. Vectors will be extracted using DINOv2 Large (1024d) and indexed instantly.")
    
    add_col1, add_col2 = st.columns([1, 1])
    with add_col1:
        new_file = st.file_uploader(
            "Upload Product Image",
            type=["jpg", "jpeg", "png", "webp", "avif"],
            key="tab3_new_file"
        )
        category_options = ["ring", "necklace", "earring", "bangle", "bracelet", "pendant", "tanmaniya", "custom"]
        chosen_cat = st.selectbox("Product Category", category_options)
        if chosen_cat == "custom":
            chosen_cat = st.text_input("Enter Custom Category Name", value="jewellery").strip()
            
        custom_sku = st.text_input("Product SKU / Code", placeholder="e.g. DRN00999 or mesh_ring_01").strip()
        
    with add_col2:
        if new_file:
            st.markdown("#### Preview New Item")
            st.image(new_file, use_container_width=True)
            
            if not custom_sku:
                default_name = os.path.splitext(new_file.name)[0]
                custom_sku = default_name
                
            if st.button("💾 Save & Index with DINOv2 Large (1024d)", type="primary", use_container_width=True):
                dest_dir = os.path.join(DATASET_PATH, "image")
                os.makedirs(dest_dir, exist_ok=True)
                
                file_ext = os.path.splitext(new_file.name)[1] or ".jpg"
                dest_filename = f"{custom_sku}{file_ext}"
                dest_path = os.path.join(dest_dir, dest_filename)
                
                with open(dest_path, "wb") as f:
                    f.write(new_file.getvalue())
                    
                st.info(f"📁 Saved file to `{dest_path}`. Generating 1024-dim DINOv2 Large vector...")
                
                item_info = {
                    "path": os.path.relpath(dest_path, os.getcwd()),
                    "abs_path": os.path.abspath(dest_path),
                    "sku": custom_sku,
                    "category": chosen_cat,
                    "mtime": time.time()
                }
                
                sync_res = sync_new_items_to_qdrant(q_client, [item_info])
                if not sync_res["errors"]:
                    st.success(f"🎉 **{custom_sku}** successfully added and indexed with DINOv2 Large (1024d)!")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"Sync error: {sync_res['errors']}")
