import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍",
    layout="centered"
)

st.title("✍ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and click **Predict**.")

# ---------------------------------------------------
# Load CNN Model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mnist_cnn_model.keras")
    return model

model = load_model()

# ---------------------------------------------------
# Drawing Canvas
# ---------------------------------------------------

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------------------------------
# Predict Button
# ---------------------------------------------------

if st.button("Predict Digit"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:

        # Convert RGBA image to RGB
        img = canvas_result.image_data.astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Resize to MNIST size
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img.astype("float32") / 255.0

        # Reshape
        img_input = img.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(img_input, verbose=0)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader(f"Predicted Digit : {digit}")

        st.success(f"Confidence : {confidence:.2f}%")

        st.image(
            img,
            caption="Processed 28 × 28 Image",
            width=150,
            clamp=True
        )

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("About")

st.sidebar.write("""
### CNN Model

- Dataset : MNIST
- Classes : 10
- Image Size : 28 × 28
- Deep Learning : CNN
- Framework : TensorFlow / Keras

### Instructions

1. Draw one digit.
2. Click Predict.
3. View prediction.
""")