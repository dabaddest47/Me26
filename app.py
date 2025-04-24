
import streamlit as st
from utils import reveal_text_under_blue, extract_text_from_image
from PIL import Image
import numpy as np

st.title("Reveal Text Under Blue Shading")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    np_image = np.array(image)
    processed_image = reveal_text_under_blue(np_image)
    st.image(processed_image, caption="Blue Shading Removed", use_column_width=True)

    extracted_text = extract_text_from_image(processed_image)
    st.subheader("Extracted Text:")
    st.text(extracted_text)
