import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio
import io
import plotly.graph_objects as go
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Forest Cover Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Caching ---
# @st.cache_resource decorator caches the model so it only loads once per session.
# This is crucial for performance and preventing repeated OOM errors on the model itself.
@st.cache_resource
def load_my_model(model_path):
    """Loads the Keras model from the specified path."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        return None

# --- Main Analysis Function ---
def analyze_forest_cover_enhanced(image_bytes, model, patch_size, threshold, progress_bar, status_text):
    """
    Processes a satellite image to calculate forest cover using a custom,
    memory-efficient chunking method.
    """
    try:
        with rasterio.open(image_bytes) as src:
            meta = src.meta
            nodata_value = src.nodata if src.nodata is not None else 0
            full_prediction_map = np.zeros((src.height, src.width), dtype=np.uint8)

            # Define a custom chunk size (in pixels) for memory efficiency.
            # This is the most important parameter to adjust for OOM errors.
            CHUNK_SIZE = 1024 
            
            num_chunks_y = (src.height + CHUNK_SIZE - 1) // CHUNK_SIZE
            num_chunks_x = (src.width + CHUNK_SIZE - 1) // CHUNK_SIZE
            total_chunks = num_chunks_y * num_chunks_x
            current_chunk_idx = 0
            
            total_valid_pixels = 0
            forest_pixels_count = 0

            # Define patch batch size for prediction
            BATCH_SIZE = 256
            
            for i in range(num_chunks_y):
                for j in range(num_chunks_x):
                    current_chunk_idx += 1
                    progress_bar.progress(current_chunk_idx / total_chunks)
                    status_text.text(f"Processing chunk {current_chunk_idx}/{total_chunks}...")

                    # Calculate the window for the current chunk
                    row_start = i * CHUNK_SIZE
                    row_end = min((i + 1) * CHUNK_SIZE, src.height)
                    col_start = j * CHUNK_SIZE
                    col_end = min((j + 1) * CHUNK_SIZE, src.width)
                    window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)

                    # Read a small chunk of data with padding for edge patches
                    pad_width = patch_size // 2
                    padded_window = window.get_buffered_window(pad_width)
                    
                    # Ensure padded window doesn't go outside the image bounds
                    padded_window = padded_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    
                    padded_chunk_data = src.read(window=padded_window)
                    padded_chunk_data_preprocessed = np.clip(padded_chunk_data, 0, 4000) / 4000.0
                    padded_chunk_data_preprocessed = padded_chunk_data_preprocessed.astype(np.float32)
                    padded_chunk_data_preprocessed = np.moveaxis(padded_chunk_data_preprocessed, 0, -1)

                    chunk_h, chunk_w = window.height, window.width
                    
                    # Create patches for the current chunk and make predictions
                    patches = []
                    y_indices, x_indices = [], []
                    
                    # Calculate the start of the non-padded region within the padded chunk
                    start_y = window.row_off - padded_window.row_off
                    start_x = window.col_off - padded_window.col_off

                    for y in range(chunk_h):
                        for x in range(chunk_w):
                            y_pad = y + start_y
                            x_pad = x + start_x
                            
                            patch = padded_chunk_data_preprocessed[y_pad:y_pad + patch_size, 
                                                                   x_pad:x_pad + patch_size, :]
                            patches.append(patch)
                            y_indices.append(y)
                            x_indices.append(x)
                            
                            # Process patches in batches
                            if len(patches) == BATCH_SIZE or (y == chunk_h - 1 and x == chunk_w - 1):
                                if not patches: continue

                                patches_np = np.array(patches)
                                predictions = model.predict(patches_np, verbose=0)
                                class_labels = (predictions > threshold).astype(np.uint8)
                                
                                # Assign predictions to the correct position in the final map
                                for k, (y_pos, x_pos) in enumerate(zip(y_indices, x_indices)):
                                    full_prediction_map[row_start + y_pos, col_start + x_pos] = class_labels[k]

                                patches = []
                                y_indices, x_indices = [], []

            # Final calculations after processing the entire image
            first_band = src.read(1)
            valid_data_mask = (first_band != nodata_value).astype(np.uint8)
            total_valid_pixels = np.sum(valid_data_mask)
            final_forest_pixels = np.sum(full_prediction_map * valid_data_mask)
            
            return final_forest_pixels, total_valid_pixels, meta

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}", icon="🚨")
        return None, None, None

# --- Streamlit UI ---
st.title("Interactive Forest Cover Analysis Dashboard")
st.markdown("Upload a correctly projected GeoTIFF file to begin.")

uploaded_file = st.file_uploader("Choose a GeoTIFF (.tif) file...", type=["tif", "tiff"])

if uploaded_file:
    # Sidebar for parameters
    with st.sidebar:
        st.title("🌳 Forest Cover AI")
        st.markdown("---")
        st.info("An interactive dashboard to analyze forest cover from GeoTIFF satellite images using a CNN.")
        st.header("Analysis Parameters")
        PATCH_SIZE = st.number_input("Model Patch Size", min_value=16, max_value=256, value=32, step=16)
        
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                                         help="Set the confidence level to classify a pixel as 'forest'. Higher values are stricter.")
        st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Results Dashboard", "⚙️ Calculation Details"])

    with tab1:
        st.subheader("Analysis Summary")
        summary_display = st.empty()
        summary_display.info("Adjust parameters in the sidebar and click 'Run Analysis'.")

    if st.button("Run Analysis", type="primary"):
        # Load the model from the same directory as the script.
        # Make sure 'forest_cover_cnn_model.keras' is in the same folder.
        model = load_my_model('forest_cover_cnn_model.keras')
        
        if model:
            image_bytes = io.BytesIO(uploaded_file.getvalue())
            
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = st.empty()

            # Call the new, enhanced analysis function
            forest_pixels, total_valid_pixels, meta = analyze_forest_cover_enhanced(
                image_bytes, model, PATCH_SIZE, confidence_threshold, progress_bar, status_text
            )
            
            progress_container.empty()
            status_text.success("Analysis Complete!")

            if forest_pixels is not None:
                # Calculations
                pixel_width = meta['transform'][0]
                pixel_height = abs(meta['transform'][4])
                area_per_pixel_m2 = pixel_width * pixel_height
                total_forest_area_km2 = (forest_pixels * area_per_pixel_m2) / 1_000_000
                total_image_area_km2 = (total_valid_pixels * area_per_pixel_m2) / 1_000_000
                non_forest_area_km2 = total_image_area_km2 - total_forest_area_km2
                forest_percentage = (forest_pixels / total_valid_pixels * 100) if total_valid_pixels > 0 else 0
                non_forest_pixels = total_valid_pixels - forest_pixels
                
                with summary_display.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Total Forest Cover", value=f"{total_forest_area_km2:.2f} km²")
                    with col2:
                        st.metric(label="Forest Percentage", value=f"{forest_percentage:.2f}%")
                    
                    st.markdown("---")
                    st.subheader("Pixel Count Verification")
                    st.markdown(f"**Forest Pixels:** `{forest_pixels:,}`")
                    st.markdown(f"**Non-Forest Pixels:** `{non_forest_pixels:,}`")
                    st.markdown("---")
                    
                    fig = go.Figure(data=[go.Pie(labels=['Forest', 'Non-Forest'], values=[forest_pixels, non_forest_pixels], hole=.5, marker_colors=['#00FF00', 'grey'])])
                    fig.update_layout(title_text='Land Cover Proportions by Pixel Count', showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.header("Calculation & CRS Details")
                    st.write(f"**Total Forest Pixels Found:** `{forest_pixels:,}`")
                    st.write(f"**Total Pixels in District Area:** `{total_valid_pixels:,}`")
                    st.write(f"**Coordinate Reference System (CRS):** `{meta['crs']}`")
                    st.write(f"**Is Geographic (units in degrees):** `{meta['crs'].is_geographic}`")
                    st.write(f"**Pixel Resolution:** `{pixel_width:.2f}m x {pixel_height:.2f}m`")
