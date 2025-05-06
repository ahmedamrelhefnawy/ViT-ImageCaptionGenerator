import streamlit as st

st.set_page_config(page_title="Image Caption Generator", page_icon="https://cdn-icons-png.flaticon.com/512/18561/18561814.png")

if 'preprocess_image' not in st.session_state:
    with st.spinner("Loading Preprocessing Module..."):
        from Services.preprocessing import preprocess_image
        st.session_state.preprocess_image = preprocess_image

if 'generate_caption' not in st.session_state:
    with st.spinner("Loading Inference Module..."):
        from Services.inference import generate_caption
        st.session_state.generate_caption = generate_caption

if "caption_model" not in st.session_state or "tokenizer" not in st.session_state:
    with st.spinner("Loading Caption Generator..."):
        from Services.loader import caption_model, tokenizer
        st.session_state.caption_model = caption_model
        st.session_state.tokenizer = tokenizer

caption_model = st.session_state.caption_model
tokenizer = st.session_state.tokenizer
preprocess_image = st.session_state.preprocess_image
generate_caption = st.session_state.generate_caption

st.title("ViT Image Caption Generator")
st.markdown(
    "This application generates captions for images using a Vision Transformer model.<br>"
    "Upload an image and click the button to generate a caption.",
    unsafe_allow_html=True
)
file = st.file_uploader("Upload File", accept_multiple_files=False)
generate_button = st.button("Generate Caption", use_container_width=True)

if file:
    st.markdown("## Loaded Image")
    st.image(file, use_container_width=False, width=400)


if generate_button:
    if not file:
        st.error("Upload an image to generate caption")
    else:
        with st.spinner("Generating Caption ..."):
            preprocessed_image = preprocess_image(file)
            caption = generate_caption(caption_model, preprocessed_image, tokenizer)

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
            <h3 style="color: #4CAF50;">Generated Caption:</h3>
            <p style="font-size: 18px;">{caption.capitalize()}</p>
            </div>
            """,
            unsafe_allow_html=True
        )