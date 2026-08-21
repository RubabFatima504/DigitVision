import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digit Vision | CNN Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Hero header */
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7F5AF0, #2CB1BC, #7F5AF0);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        animation: shine 4s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    .hero-subtitle {
        text-align: center;
        color: #B4B4C7;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Card container */
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        margin-bottom: 1rem;
    }

    .card-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: #F1F1F6;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Prediction number */
    .prediction-number {
        font-family: 'Poppins', sans-serif;
        font-size: 5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #7F5AF0, #2CB1BC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .confidence-badge {
        text-align: center;
        color: #2CB1BC;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    /* Metric pills */
    .metric-pill {
        background: rgba(127, 90, 240, 0.15);
        border: 1px solid rgba(127, 90, 240, 0.35);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .metric-pill .value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F1F1F6;
    }
    .metric-pill .label {
        color: #B4B4C7;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1730 0%, #24243e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .stButton>button {
        background: linear-gradient(135deg, #7F5AF0, #2CB1BC);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(127, 90, 240, 0.4);
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn.keras")
    return model


# ---------------------------------------------------------
# Preprocessing: canvas RGBA -> normalized 28x28x1
# ---------------------------------------------------------
def preprocess_canvas_image(image_data):
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # canvas is black bg white stroke -> MNIST wants white digit on... handled below
    img = ImageOps.invert(img)  # keep as drawn (white stroke on black), matches MNIST style

    arr = np.array(img)

    # Crop to bounding box of the digit for better centering
    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        return None, None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]

    # Pad to square
    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = np.pad(
        cropped,
        ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)),
        mode="constant",
        constant_values=0,
    )

    # Add margin like MNIST (digits aren't edge-to-edge)
    margin = int(size * 0.2)
    padded = np.pad(padded, margin, mode="constant", constant_values=0)

    img_final = Image.fromarray(padded).resize((28, 28), Image.LANCZOS)
    arr_final = np.array(img_final).astype("float32") / 255.0

    display_img = img_final
    model_input = arr_final.reshape(1, 28, 28, 1)
    return model_input, display_img


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    arr = np.array(img)

    # Auto-detect polarity: MNIST digits are white-on-black
    if arr.mean() > 127:
        arr = 255 - arr

    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        return None, None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = np.pad(
        cropped,
        ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)),
        mode="constant",
        constant_values=0,
    )
    margin = int(size * 0.2)
    padded = np.pad(padded, margin, mode="constant", constant_values=0)

    img_final = Image.fromarray(padded).resize((28, 28), Image.LANCZOS)
    arr_final = np.array(img_final).astype("float32") / 255.0

    return arr_final.reshape(1, 28, 28, 1), img_final


# ---------------------------------------------------------
# Confidence chart
# ---------------------------------------------------------
def build_confidence_chart(probs):
    digits = list(range(10))
    colors = ["#2CB1BC" if p != max(probs) else "#7F5AF0" for p in probs]

    fig = go.Figure(
        go.Bar(
            x=digits,
            y=probs * 100,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="#F1F1F6", size=11),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B4B4C7", family="Inter"),
        xaxis=dict(
            title="Digit",
            tickmode="array",
            tickvals=digits,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Confidence (%)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.05)",
        ),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔢 Digit Vision")
    st.markdown("A CNN-powered handwritten digit classifier trained on MNIST.")
    st.markdown("---")

    st.markdown("#### 📊 Model Performance")
    st.markdown(
        """
        <div class="metric-pill">
            <div class="value">99.04%</div>
            <div class="label">Test Accuracy</div>
        </div>
        <div class="metric-pill">
            <div class="value">0.0305</div>
            <div class="label">Test Loss</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🏗️ Architecture")
    st.markdown(
        """
        - Conv2D(32) → MaxPool
        - Conv2D(64) → MaxPool
        - Flatten → Dense(64)
        - Dense(10, softmax)
        - **121,930** parameters
        """
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    input_mode = st.radio("Input method", ["✍️ Draw a digit", "📤 Upload an image"])
    stroke_width = st.slider("Brush size", 8, 30, 18) if input_mode == "✍️ Draw a digit" else None

    st.markdown("---")
    st.caption("Built with TensorFlow & Streamlit")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="hero-title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Draw a digit or upload an image — a CNN trained on MNIST will predict it in real time.</div>',
    unsafe_allow_html=True,
)

model = load_model()

col1, col2 = st.columns([1, 1], gap="large")

model_input = None
display_img = None

# ---------------------------------------------------------
# Left column: Input
# ---------------------------------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if input_mode == "✍️ Draw a digit":
        st.markdown('<div class="card-title">✍️ Draw here</div>', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        predict_clicked = st.button("🔮 Predict")
        clear_hint = st.caption("Use the toolbar (bottom-left of canvas) to erase or undo.")

        if canvas_result.image_data is not None and canvas_result.image_data[..., :3].sum() > 0:
            model_input, display_img = preprocess_canvas_image(canvas_result.image_data)
    else:
        st.markdown('<div class="card-title">📤 Upload a digit image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PNG or JPG file", type=["png", "jpg", "jpeg"])
        predict_clicked = st.button("🔮 Predict")
        if uploaded_file is not None:
            model_input, display_img = preprocess_uploaded_image(uploaded_file)
            st.image(uploaded_file, caption="Original upload", width=200)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Right column: Prediction
# ---------------------------------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Prediction</div>', unsafe_allow_html=True)

    if model_input is not None:
        preds = model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        pcol1, pcol2 = st.columns([1, 1])
        with pcol1:
            st.markdown(f'<div class="prediction-number">{predicted_digit}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-badge">{confidence:.1f}% confident</div>', unsafe_allow_html=True)
        with pcol2:
            if display_img is not None:
                st.image(display_img.resize((140, 140)), caption="Model input (28×28)")

        st.markdown("##### Confidence across all digits")
        st.plotly_chart(build_confidence_chart(preds), use_container_width=True)
    else:
        st.info("Draw a digit or upload an image, then click **Predict** to see results here.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center; color:#6b6b83; margin-top:2rem; font-size:0.85rem;">'
    "Model trained on the MNIST dataset · CNN built with TensorFlow/Keras"
    "</p>",
    unsafe_allow_html=True,
)