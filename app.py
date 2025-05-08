import streamlit as st
import time

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
        from Services.loader import encoder, decoder, tokenizer
        st.session_state.encoder = encoder
        st.session_state.decoder = decoder
        st.session_state.tokenizer = tokenizer

st.session_state.encoder = encoder
st.session_state.decoder = decoder
tokenizer = st.session_state.tokenizer
preprocess_image = st.session_state.preprocess_image
generate_caption = st.session_state.generate_caption

st.title("ViT Image Caption Generator")
st.markdown(
    "This application generates captions for images using a Vision Transformer model.<br>"
    "Upload an image and click the button to generate a caption.",
    unsafe_allow_html=True
)
valid_formats = ["jpg", "jpeg", "png"]
file = st.file_uploader("Upload File", accept_multiple_files=False, type= valid_formats, label_visibility="collapsed")

if file:
    file_name = file.name
    file_extension = file_name.split(".")[-1]
    
    if file_extension not in valid_formats:
        st.error(file_extension, "is not a valid image format. Please upload a jpg, jpeg, or png file.")
    else:
        generate_button = st.button("Generate Caption", use_container_width=True)

        st.markdown("## Loaded Image")
        st.image(file, use_container_width=False, width=400)

        if generate_button:
            if not file:
                st.error("Upload an image to generate caption")
            else:
                with st.spinner("Generating Caption ..."):
                    preprocessed_image = preprocess_image(file)
                    start = time.time()
                    caption = generate_caption(encoder, decoder, preprocessed_image, tokenizer)
                    end = time.time()

                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 20px;">
                    <h3 style="color: #4CAF50;">Generated Caption:</h3>
                    <p style="font-size: 18px;">{caption.capitalize()}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.text(f"Time taken: {end - start:.2f} seconds")

developers_info = {
    "Ahmed Adel Mohammed": {
        "GitHub": "https://github.com/ahmeda335",
        "LinkedIn": "www.linkedin.com/in/ahmed-adel-b748aa23b",
        "Contact": "ahmedadel30320032003@gmail.com"
    },
    "Ahmed Eltokhy": {
        "GitHub": "https://github.com/ahmdeltoky03",
        "LinkedIn": "https://www.linkedin.com/in/ahmd-eltokhey-8577b3275",
        "Contact": "ahmdeltoky4@gmail.com"
    },
    "Amr Ghanem": {
        "GitHub": "https://github.com/AmrGhanem13",
        "LinkedIn": "https://www.linkedin.com/in/amr-ghanem-306b392b9",
        "Contact": "Amr_Ghanem07@yahoo.com"
    },
    "Samir Mohammed": {
        "GitHub": "https://github.com/samir-m0hamed",
        "LinkedIn": "https://www.linkedin.com/in/samir-mohamed-2976bb252",
        "Contact": "samirmohamed122003@gmail.com"
    },
    "Ahmed Elhefnawy": {
        "GitHub": "https://github.com/ahmedamrelhefnawy",
        "LinkedIn": "https://www.linkedin.com/in/ahmed-elhefnawy-258949243/",
        "Contact": "ahmedelhefnawy2003@hotmail.com"
    }
}

readme_format = []
for name, info in developers_info.items():  
    readme_format.append(f"- {name}:")
    readme_format[-1]+=(f"  ([LinkedIn]({info['LinkedIn']}))" if info['LinkedIn'] else "(LinkedIn: Not Available")
    readme_format[-1]+=(f"  ([GitHub]({info['GitHub']}))")
    readme_format[-1]+=(f"  ([Contact]({info['Contact']}))")

footer = '\n'.join(readme_format)
footer = f'''<small style="color:gray;">Developed with ❤️ by:<br>\n{footer}\n</small>'''

st.markdown("---")
st.markdown(footer, unsafe_allow_html=True)