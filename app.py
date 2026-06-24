import streamlit as st
import torch
import json
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from src.model import InkTraceModel

# Setup
st.set_page_config(
    page_title="InkTrace OCR",
    layout="wide"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load vocab
@st.cache_resource
def load_vocab():
    with open("char_map.json", "r") as f:
        vocab = json.load(f)

    idx2char = {v: k for k, v in vocab.items()}
    return vocab, idx2char

# Load model
@st.cache_resource
def load_model():

    vocab, _ = load_vocab()

    model = InkTraceModel(
        vocab_size=len(vocab)
    ).to(device)

    # Load models accordingly
    model.load_state_dict(
        torch.load(
            "checkpoints/inktrace_epoch_50.pth",
            map_location=device
        )
    )

    model.eval()

    return model


vocab, idx2char = load_vocab()
model = load_model()

# Image Processing
def process_image(img):

    img = img.convert("L")

    w, h = img.size

    new_w = int(w * (64 / h))

    img = img.resize(
        (new_w, 64),
        Image.Resampling.LANCZOS
    )

    target_w = 512

    if new_w < target_w:
        pad_width = target_w - new_w

        img = ImageOps.expand(
            img,
            border=(0, 0, pad_width, 0),
            fill=255
        )

    else:
        img = img.crop((0, 0, target_w, 64))

    tensor = transforms.ToTensor()(img)

    tensor = transforms.Normalize(
        (0.5,),
        (0.5,)
    )(tensor)

    return tensor.unsqueeze(0).to(device)

# OCR
def predict_image(img):

    tensor = process_image(img)

    with torch.no_grad():
        output = model(tensor)

    pred_indices = torch.argmax(
        output,
        dim=2
    ).squeeze(0)

    decoded = []
    last_idx = -1

    for idx in pred_indices:

        idx = idx.item()

        if idx != 0 and idx != last_idx:
            decoded.append(idx)

        last_idx = idx

    chars = [
        idx2char.get(i, "?")
        for i in decoded
    ]

    text = "".join(chars)

    return text, decoded


# UI
st.title("InkTrace OCR")

st.write(
    "Upload a handwritten image and run OCR."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Input",
            use_container_width=True
        )

    if st.button("Run OCR"):

        text, indices = predict_image(image)

        with col2:

            st.subheader("Prediction")

            st.code(text)

            with st.expander(
                "Decoded Indices"
            ):
                st.write(indices)