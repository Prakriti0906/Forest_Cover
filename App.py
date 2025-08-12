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
def analyze_forest_cover(image_bytes, model, patch_size, threshold, progress_bar, status_text):
    """
    Processes a satellite image to calculate forest cover in memory-efficient batches.
    This version includes nested batching to prevent memory issues with large blocks.
    """
    try:
        with rasterio.open(image_bytes) as src:
            meta = src.meta
            nodata_value = src.nodata if src.nodata is not None else 0
            first_band = src.read(1)
            valid_data_mask = (first_band != nodata_value).astype(np.uint8)
            total_valid_pixels = np.sum(valid_data_mask)

            full_prediction_map = np.zeros((src.height, src.width), dtype=np.uint8)
            num_blocks = len(list(src.block_windows(1)))
            current_block = 0
            
            # Define a smaller batch size for processing patches
            BATCH_SIZE = 256 # You can adjust this value based on your available memory

            for ji, window in src.block_windows(1):
                current_block += 1
                progress_bar.progress(current_block / num_blocks)
                status_text.text(f"Processing block {current_block}/{num_blocks}...")

                block_data = src.read(window=window)
                block_data_preprocessed = np.clip(block_data, 0, 4000) / 4000.0
                block_data_preprocessed = block_data_preprocessed.astype(np.float32)
                block_data_preprocessed = np.moveaxis(block_data_preprocessed, 0, -1)
                
                pad_width = patch_size // 2
                padded_block = np.pad(block_data_preprocessed, 
                                      [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], 
                                      mode='reflect')
                
                block_h, block_w, _ = block_data_preprocessed.shape
                block_predictions = np.zeros((block_h, block_w), dtype=np.uint8)

                patches = []
                patch_positions = []
                for y in range(block_h):
                    for x in range(block_w):
                        patches.append(padded_block[y: y + patch_size, x: x + patch_size, :])
                        patch_positions.append((y, x))
                        
                        # Process patches in batches instead of all at once
                        if len(patches) == BATCH_SIZE:
                            patches_np = np.array(patches)
                            predictions = model.predict(patches_np, verbose=0)
                            class_labels = (predictions > threshold).astype(np.uint8)

                            for (y_pos, x_pos), label in zip(patch_positions, class_labels):
                                block_predictions[y_pos, x_pos] = label

                            patches = [] # Clear patches list for the next batch
                            patch_positions = [] # Clear positions list
                
                # Process any remaining patches in the last batch
                if patches:
                    patches_np = np.array(patches)
                    predictions = model.predict(patches_np, verbose=0)
                    class_labels = (predictions > threshold).astype(np.uint8)
                    for (y_pos, x_pos), label in zip(patch_positions, class_labels):
                        block_predictions[y_pos, x_pos] = label

                full_prediction_map[window.row_off:window.row_off + window.height, 
                                    window.col_off:window.col_off + window.width] = block_predictions
            
            final_forest_pixels = np.sum(full_prediction_map * valid_data_mask)
            return final_forest_pixels, total_valid_pixels, meta

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}", icon="🚨")
        return None, None, None

# --- Streamlit UI ---
with st.sidebar:
    st.title("🌳 Forest Cover AI")
    st.markdown("---")
    st.info("An interactive dashboard to analyze forest cover from GeoTIFF satellite images using a CNN.")
    st.header("Analysis Parameters")
    PATCH_SIZE = st.number_input("Model Patch Size", min_value=16, max_value=256, value=32, step=16)
    
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                                     help="Set the confidence level to classify a pixel as 'forest'. Higher values are stricter.")
    st.markdown("---")


# --- Main Page ---
st.title("Interactive Forest Cover Analysis Dashboard")
st.markdown("Upload a correctly projected GeoTIFF file to begin.")
uploaded_file = st.file_uploader("Choose a GeoTIFF (.tif) file...", type=["tif", "tiff"])

if uploaded_file is not None:
    tab1, tab2 = st.tabs(["📊 Results Dashboard", "⚙️ Calculation Details"])

    with tab1:
        st.subheader("Analysis Summary")
        summary_display = st.empty()
        summary_display.info("Adjust parameters in the sidebar and click 'Run Analysis'.")

    if st.button("Run Analysis", type="primary"):
        model = load_my_model('forest_cover_cnn_model.keras')
        if model:
            image_bytes = io.BytesIO(uploaded_file.getvalue())
            
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = st.empty()

            forest_pixels, total_valid_pixels, meta = analyze_forest_cover(
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