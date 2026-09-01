import streamlit as st
import requests
import io
import os
import time
import base64
from PIL import Image
from qdrant_client import QdrantClient
from sync_manager import check_catalog_sync_status, sync_new_items_to_qdrant, scan_disk_catalog

# Page configuration
st.set_page_config(
    page_title="AI Jewellery Visual Search & Model Benchmark",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Executive Dashboard Aesthetics
st.markdown("""
<style>
    /* Dark glassmorphic modern background */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header title styling */
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
        margin-bottom: 1.5rem;
    }
    
    /* Card container */
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
        border-color: rgba(212, 175, 55, 0.7);
    }
    
    /* Similarity badge */
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
    
    .model-badge {
        background: #334155;
        color: #cbd5e1;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration Constants
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000/embed")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

MODELS_CONFIG = {
    "clip-base": {
        "label": "CLIP ViT-B/32 (Semantic Baseline)",
        "collection": "inv_clip_base",
        "desc": "General visual semantics (category, gold tone, background)"
    },
    "clip-large": {
        "label": "CLIP ViT-L/14 (High-Res CLIP)",
        "collection": "inv_clip_large",
        "desc": "Higher resolution semantic feature matching"
    },
    "dinov2-base": {
        "label": "DINOv2 Base (Micro-Detail Specialist - 768d)",
        "collection": "inv_dinov2_base",
        "desc": "Dense spatial patch features & local texture geometry"
    },
    "dinov2-large": {
        "label": "DINOv2 Large (Micro-Detail Specialist - 1024d)",
        "collection": "inv_dinov2_large",
        "desc": "Maximum spatial sensitivity for fine grid/mesh/hammered details"
    }
}

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


def embed_image_api(image_bytes: bytes, model_name: str, remove_bg: bool) -> dict:
    files = {"file": ("query.jpg", image_bytes, "image/jpeg")}
    params = {"model_name": model_name, "remove_bg": str(remove_bg).lower()}
    response = requests.post(INFERENCE_API_URL, params=params, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def search_qdrant(client: QdrantClient, collection_name: str, vector: list[float], top_k: int = 5):
    try:
        existing_cols = [c.name for c in client.get_collections().collections]
        # If model collection is empty but default inventory exists and matches dimension (512)
        target_col = collection_name
        if target_col not in existing_cols and "inventory" in existing_cols and len(vector) == 512:
            target_col = "inventory"

        if target_col not in existing_cols:
            return []

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=target_col,
                query=vector,
                limit=top_k,
                with_payload=True
            )
            return response.points
        elif hasattr(client, "search"):
            return client.search(
                collection_name=target_col,
                query_vector=vector,
                limit=top_k,
                with_payload=True
            )
        else:
            return []
    except Exception as e:
        st.error(f"Qdrant Search Error ({collection_name}): {e}")
        return []


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/diamond.png", width=64)
st.sidebar.title("Search Controls")

selected_model_key = st.sidebar.selectbox(
    "Select Embedding Model",
    options=list(MODELS_CONFIG.keys()),
    format_func=lambda k: MODELS_CONFIG[k]["label"]
)

remove_bg_toggle = st.sidebar.checkbox(
    "🧹 Remove Background / Noise",
    value=True,
    help="Applies AI Background Isolation (rembg) to isolate the jewellery piece and filter out background reflections, skin, and cloth noise."
)

top_k_slider = st.sidebar.slider("Top Results Count", min_value=3, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Catalog Sync Manager")
q_client = get_qdrant_client()

# Check folder vs vector DB sync status
sync_status = check_catalog_sync_status(q_client)
total_disk = sync_status["total_disk_items"]
has_unindexed = sync_status["has_unindexed"]
new_items_list = sync_status["new_items"]

if has_unindexed:
    st.sidebar.warning(f"🔔 **{len(new_items_list)} New Image(s) Detected!**")
    if st.sidebar.button("⚡ Sync Catalog Now", type="primary", use_container_width=True):
        with st.spinner("Indexing new jewellery into vector collections..."):
            res = sync_new_items_to_qdrant(q_client, new_items_list)
            st.sidebar.success(f"✅ Synced {res['synced_count']} items!")
            time.sleep(1)
            st.rerun()
else:
    st.sidebar.success(f"✅ All {total_disk} items indexed & in-sync")

if st.sidebar.button("🔄 Scan Folder for New Files", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### System Health")
try:
    cols = [c.name for c in q_client.get_collections().collections]
    st.sidebar.success(f"⚡ Qdrant Online ({len(cols)} Collections)")
except Exception as e:
    st.sidebar.error("❌ Qdrant Database Offline")

try:
    r_health = requests.get("http://localhost:8000/docs", timeout=2)
    if r_health.status_code == 200:
        st.sidebar.success("⚡ Inference API Online")
except Exception:
    st.sidebar.error("❌ Inference API Offline")

# Header Section
st.markdown('<div class="main-header">💎 AI Jewellery Visual Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fine Micro-Detail & Texture Pattern Matcher (Grid Mesh vs. Hammered vs. Filigree)</div>', unsafe_allow_html=True)

# Notification Banner for New Unindexed Images
if has_unindexed:
    st.warning(f"🔔 **Catalog Update**: {len(new_items_list)} new jewellery image(s) detected in `./Jewellery_Data/`!")
    with st.expander(f"👁️ View {len(new_items_list)} Unindexed Images & Quick Sync", expanded=True):
        preview_count = min(len(new_items_list), 6)
        sync_cols = st.columns(preview_count)
        for idx, item in enumerate(new_items_list[:preview_count]):
            with sync_cols[idx]:
                if os.path.exists(item["path"]):
                    st.image(item["path"], use_container_width=True)
                st.caption(f"**{item['sku']}**\n\nCategory: `{item['category']}`")
                
        if st.button("⚡ Index & Sync All New Images Across 4 Models", type="primary", use_container_width=True):
            pbar = st.progress(0.0)
            status_placeholder = st.empty()
            
            def handle_prog(cur, tot, desc):
                pbar.progress(cur / tot)
                status_placeholder.text(f"[{cur}/{tot}] {desc}")
                
            res = sync_new_items_to_qdrant(q_client, new_items_list, progress_callback=handle_prog)
            st.success(f"🎉 Successfully indexed {res['synced_count']} new items across all 4 vector models!")
            time.sleep(1.5)
            st.rerun()

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Model Search",
    "📊 Multi-Model Side-by-Side Benchmark",
    "➕ Add & Sync New Jewellery"
])

with tab1:
    col_upload, col_preview = st.columns([1, 1])
    
    with col_upload:
        st.markdown("### 1. Provide Query Image")
        uploaded_file = st.file_uploader("Upload Ring or Necklace Image", type=["jpg", "jpeg", "png", "webp", "avif"])

        
        sample_choice = st.selectbox(
            "Or choose from test gallery:",
            [
                "(None)",
                "Model Wearing Gold Ring (test/images.jpeg)",
                "Grid Mesh Ring (test/shopping.jpeg)",
                "Gold Platinum Ring (test/gold-platinum-ring.avif)",
                "Sample Ring 081 (Jewellery_Data/ring/ring_081.jpg)"
            ]
        )

    # Determine query image bytes
    query_bytes = None
    query_name = "Uploaded Image"
    
    if uploaded_file is not None:
        query_bytes = uploaded_file.getvalue()
        query_name = uploaded_file.name
    elif sample_choice != "(None)":
        if "images.jpeg" in sample_choice:
            img_path = "./test/images.jpeg"
        elif "shopping.jpeg" in sample_choice:
            img_path = "./test/shopping.jpeg"
        elif "gold-platinum" in sample_choice:
            img_path = "./test/gold-platinum-ring-377062544-zkx5j.jpg.avif"
        else:
            img_path = "./Jewellery_Data/ring/ring_081.jpg"
        
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
                    buf = io.BytesIO()
                    cropped_pil.save(buf, format="JPEG")
                    effective_query_bytes = buf.getvalue()
                except Exception as e:
                    st.warning(f"Cropper tool warning: {e}")

    with col_preview:
        if query_bytes:
            st.markdown("### 2. Input Image Previews")
            p_col1, p_col2 = st.columns(2)
            show_input = Image.open(io.BytesIO(effective_query_bytes))
            p_col1.image(show_input, caption="Target Query Image (Focused)", use_container_width=True)
            
            if remove_bg_toggle:
                st.info("AI Background Noise Removal & Auto-Zoom Enabled (Isolating Jewellery...)")
            else:
                st.warning("Background Noise Removal Disabled")

    if effective_query_bytes and st.button("🚀 Run Visual Similarity Search", type="primary", use_container_width=True):
        model_info = MODELS_CONFIG[selected_model_key]
        with st.spinner(f"Extracting vectors using {model_info['label']}..."):
            try:
                res_embed = embed_image_api(effective_query_bytes, selected_model_key, remove_bg_toggle)
                vec = res_embed["vector"]
                
                # Show processed background-removed image if available
                if "processed_image_b64" in res_embed and remove_bg_toggle:
                    proc_b64 = res_embed["processed_image_b64"]
                    proc_bytes = base64.b64decode(proc_b64)
                    with col_preview:
                        p_col2.image(Image.open(io.BytesIO(proc_bytes)), caption="AI Cleaned & Zoomed (Background Removed)", use_container_width=True)


                # Query Qdrant
                hits = search_qdrant(q_client, model_info["collection"], vec, top_k=top_k_slider)
                
                st.markdown("---")
                if len(hits) == 0:
                    st.warning(f"No items found in collection '{model_info['collection']}'. You can index the dataset for this model by clicking 'Index Dataset' in the sidebar or running benchmark.py.")
                else:
                    st.markdown(f"### Top {len(hits)} Similar Items Retrieved (`{model_info['collection']}`)")
                    num_cols = min(len(hits), 5)
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
                            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                            if rel_path and os.path.exists(rel_path):
                                st.image(rel_path, use_container_width=True)
                            else:
                                st.write("📷 [Image Preview]")
                            
                            st.markdown(f"**{sku}**")
                            st.markdown(f"<span class='model-badge'>{cat}</span>", unsafe_allow_html=True)
                            st.markdown(f"<div style='margin-top: 6px;'><span class='score-badge score-badge-high'>{score_pct:.2f}% Match</span></div>", unsafe_allow_html=True)
                            st.markdown(f"</div>", unsafe_allow_html=True)


            except Exception as e:
                st.error(f"Execution Error: {e}")

with tab2:
    st.markdown("### 📊 Side-by-Side Model Benchmark Comparison")
    st.markdown("Compare semantic focus (CLIP) vs. fine micro-texture focus (DINOv2) on the exact same query image.")
    
    if effective_query_bytes and st.button("⚡ Compare All 4 Models Simultaneously", type="secondary", use_container_width=True):
        cols = st.columns(4)
        for idx, (m_key, m_cfg) in enumerate(MODELS_CONFIG.items()):
            with cols[idx]:
                st.markdown(f"#### {m_cfg['label'].split(' ')[0]} {m_cfg['label'].split(' ')[1]}")
                st.caption(m_cfg["desc"])
                
                try:
                    res = embed_image_api(effective_query_bytes, m_key, remove_bg_toggle)
                    v = res["vector"]
                    hits = search_qdrant(q_client, m_cfg["collection"], v, top_k=5)

                    
                    for rank, h in enumerate(hits, 1):
                        p = h.payload or {}
                        sku = p.get("sku", f"ID_{h.id}")
                        path = p.get("path", "")
                        score = h.score
                        
                        st.markdown(f"**#{rank} {sku}** ({score*100:.1f}%)")
                        if path and os.path.exists(path):
                            st.image(path, use_container_width=True)
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error querying {m_key}: {e}")

with tab3:
    st.markdown("### ➕ Add New Jewellery to Catalog (Instant Vector Sync)")
    st.markdown("Upload new products directly to the catalog. Vectors will be generated and auto-synced across all 4 AI models instantly.")
    
    add_col1, add_col2 = st.columns([1, 1])
    with add_col1:
        new_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png", "webp", "avif"], key="tab3_new_file")
        category_options = ["ring", "necklace", "earring", "bracelet", "pendant", "custom"]
        chosen_cat = st.selectbox("Product Category", category_options)
        if chosen_cat == "custom":
            chosen_cat = st.text_input("Enter Custom Category Name", value="jewellery").strip()
            
        custom_sku = st.text_input("Product SKU / Code", placeholder="e.g. ring_501 or diamond_band_01").strip()
        
    with add_col2:
        if new_file:
            st.markdown("#### Preview New Item")
            st.image(new_file, use_container_width=True)
            
            if not custom_sku:
                default_name = os.path.splitext(new_file.name)[0]
                custom_sku = default_name
                
            if st.button("💾 Save to Catalog & Auto-Index All 4 Models", type="primary", use_container_width=True):
                # 1. Ensure target directory exists
                dest_dir = os.path.join("./Jewellery_Data", chosen_cat)
                os.makedirs(dest_dir, exist_ok=True)
                
                # 2. Save file
                file_ext = os.path.splitext(new_file.name)[1] or ".jpg"
                dest_filename = f"{custom_sku}{file_ext}"
                dest_path = os.path.join(dest_dir, dest_filename)
                
                with open(dest_path, "wb") as f:
                    f.write(new_file.getvalue())
                    
                st.info(f"📁 Saved file to `{dest_path}`. Extracting vectors for 4 AI models...")
                
                # 3. Trigger auto-sync
                item_info = {
                    "path": os.path.relpath(dest_path, os.getcwd()),
                    "abs_path": os.path.abspath(dest_path),
                    "sku": custom_sku,
                    "category": chosen_cat,
                    "mtime": time.time()
                }
                
                prog = st.progress(0.0)
                status_box = st.empty()
                def on_add_prog(c, t, msg):
                    prog.progress(c / t)
                    status_box.text(f"[{c}/{t}] {msg}")
                    
                sync_res = sync_new_items_to_qdrant(q_client, [item_info], progress_callback=on_add_prog)
                if not sync_res["errors"]:
                    st.success(f"🎉 **{custom_sku}** successfully added to catalog and indexed across all 4 vector models!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Sync encountered errors: {sync_res['errors']}")

