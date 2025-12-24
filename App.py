import streamlit as st
import tensorflow as tf
import numpy as np
import rasterio
import rasterio.windows
import io
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Forest Cover Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Model Caching
# ---------------------------------------------------
@st.cache_resource
def load_my_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}", icon="🚨")
        return None


# ---------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------
def analyze_forest_cover_enhanced(
    image_bytes,
    model,
    patch_size,
    threshold,
    progress_bar,
    status_text
):
    try:
        with rasterio.open(image_bytes) as src:
            meta = src.meta
            nodata_value = src.nodata if src.nodata is not None else 0

            full_prediction_map = np.zeros(
                (src.height, src.width),
                dtype=np.uint8
            )

            CHUNK_SIZE = 256
            pad_width = patch_size // 2
            BATCH_SIZE = 128

            num_chunks_y = (src.height + CHUNK_SIZE - 1) // CHUNK_SIZE
            num_chunks_x = (src.width + CHUNK_SIZE - 1) // CHUNK_SIZE
            total_chunks = num_chunks_y * num_chunks_x
            current_chunk = 0

            for i in range(num_chunks_y):
                for j in range(num_chunks_x):
                    current_chunk += 1
                    progress_bar.progress(current_chunk / total_chunks)
                    status_text.text(
                        f"Processing chunk {current_chunk}/{total_chunks}..."
                    )

                    row_start = i * CHUNK_SIZE
                    row_end = min((i + 1) * CHUNK_SIZE, src.height)
                    col_start = j * CHUNK_SIZE
                    col_end = min((j + 1) * CHUNK_SIZE, src.width)

                    window = rasterio.windows.Window(
                        col_start,
                        row_start,
                        col_end - col_start,
                        row_end - row_start
                    )

                    # -------- FIXED BUFFER WINDOW (version-safe) --------
                    padded_window = rasterio.windows.Window(
                        col_off=max(window.col_off - pad_width, 0),
                        row_off=max(window.row_off - pad_width, 0),
                        width=min(window.width + 2 * pad_width, src.width),
                        height=min(window.height + 2 * pad_width, src.height),
                    )

                    padded_window = padded_window.intersection(
                        rasterio.windows.Window(
                            0, 0, src.width, src.height
                        )
                    )

                    padded_data = src.read(window=padded_window)

                    padded_data = np.clip(padded_data, 0, 4000) / 4000.0
                    padded_data = padded_data.astype(np.float32)
                    padded_data = np.moveaxis(padded_data, 0, -1)

                    chunk_h = window.height
                    chunk_w = window.width

                    start_y = int(window.row_off - padded_window.row_off)
                    start_x = int(window.col_off - padded_window.col_off)

                    patches = []
                    positions = []

                    for y in range(chunk_h):
                        for x in range(chunk_w):
                            y_pad = y + start_y
                            x_pad = x + start_x

                            patch = padded_data[
                                y_pad:y_pad + patch_size,
                                x_pad:x_pad + patch_size,
                                :
                            ]

                            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                                continue

                            patches.append(patch)
                            positions.append((y, x))

                            if len(patches) == BATCH_SIZE:
                                preds = model.predict(
                                    np.array(patches),
                                    verbose=0
                                )
                                labels = (preds > threshold).astype(np.uint8)

                                for k, (yy, xx) in enumerate(positions):
                                    full_prediction_map[
                                        row_start + yy,
                                        col_start + xx
                                    ] = labels[k]

                                patches, positions = [], []

                    if patches:
                        preds = model.predict(
                            np.array(patches),
                            verbose=0
                        )
                        labels = (preds > threshold).astype(np.uint8)

                        for k, (yy, xx) in enumerate(positions):
                            full_prediction_map[
                                row_start + yy,
                                col_start + xx
                            ] = labels[k]

            first_band = src.read(1)
            valid_mask = (first_band != nodata_value).astype(np.uint8)

            forest_pixels = np.sum(full_prediction_map * valid_mask)
            total_valid_pixels = np.sum(valid_mask)

            return forest_pixels, total_valid_pixels, meta

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}", icon="🚨")
        return None, None, None


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🌳 Interactive Forest Cover Analysis Dashboard")
st.markdown("Upload a GeoTIFF satellite image to estimate forest cover.")

uploaded_file = st.file_uploader(
    "Choose a GeoTIFF (.tif / .tiff)",
    type=["tif", "tiff"]
)

if uploaded_file:
    with st.sidebar:
        st.header("Model Parameters")
        PATCH_SIZE = st.number_input(
            "Patch Size",
            min_value=16,
            max_value=256,
            value=32,
            step=16
        )
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.75, 0.05
        )

    tab1, tab2 = st.tabs(
        ["📊 Results Dashboard", "⚙️ Calculation Details"]
    )

    with tab1:
        summary_display = st.empty()

    if st.button("Run Analysis", type="primary"):
        model = load_my_model("forest_cover_cnn_model.keras")

        if model:
            image_bytes = io.BytesIO(uploaded_file.getvalue())

            progress_bar = st.progress(0)
            status_text = st.empty()

            forest_pixels, total_pixels, meta = analyze_forest_cover_enhanced(
                image_bytes,
                model,
                PATCH_SIZE,
                confidence_threshold,
                progress_bar,
                status_text
            )

            status_text.success("Analysis Complete!")

            if forest_pixels is not None:
                pixel_w = meta["transform"][0]
                pixel_h = abs(meta["transform"][4])
                area_per_pixel = pixel_w * pixel_h

                forest_area_km2 = (
                    forest_pixels * area_per_pixel
                ) / 1_000_000
                total_area_km2 = (
                    total_pixels * area_per_pixel
                ) / 1_000_000

                forest_pct = (
                    forest_pixels / total_pixels * 100
                    if total_pixels > 0 else 0
                )

                non_forest_pixels = total_pixels - forest_pixels

                with summary_display.container():
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Forest Area (km²)",
                        f"{forest_area_km2:.2f}"
                    )
                    col2.metric(
                        "Forest Percentage",
                        f"{forest_pct:.2f}%"
                    )

                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Forest", "Non-Forest"],
                                values=[
                                    forest_pixels,
                                    non_forest_pixels
                                ],
                                hole=0.5
                            )
                        ]
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with tab2:
                    st.write(
                        f"**Total Forest Pixels:** {forest_pixels:,}"
                    )
                    st.write(
                        f"**Total Valid Pixels:** {total_pixels:,}"
                    )
                    st.write(
                        f"**CRS:** {meta['crs']}"
                    )
                    st.write(
                        f"**Pixel Resolution:** "
                        f"{pixel_w:.2f}m x {pixel_h:.2f}m"
                    )

