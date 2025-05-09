# Vision Transformer Image Caption Generator

This project implements an **Image Caption Generator** using a Vision Transformer (ViT) model. The application takes an image as input and generates a descriptive caption for it.

## Features

- **Vision Transformer (ViT)**: Utilizes a transformer-based architecture for image processing.
- **Streamlit Interface**: Provides an interactive web-based interface for uploading images and generating captions.
- **Preprocessing and Inference**: Includes modules for image preprocessing and caption generation.

## Installation


1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser (default: `http://localhost:8501`).
2. Upload an image using the file uploader.
3. Click the "Generate Caption" button to generate a caption for the uploaded image.

## Project Structure

- **`app.py`**: Main application file for the Streamlit interface.
- **`Classes/vit_layers.py`**: Contains custom Vision Transformer layers and components.
- **`Services/`**: Includes modules for preprocessing, inference, and model loading.
- **`README.md`**: Documentation for the project.

## Requirements

- Python 3.11+ (Recommended)
- TensorFlow
- Streamlit

Install additional dependencies listed in `requirements.txt`.

## Acknowledgments

- Vision Transformer (ViT) architecture inspired by [Google Research](https://github.com/google-research/vision_transformer).
- Streamlit for providing an easy-to-use interface.

## Notebooks:

- [Visions Transformer Notebook](https://www.kaggle.com/code/ahmedadel300/vit-image-captioning)
- [CNN Notebook](https://www.kaggle.com/code/ahmedadel300/image-caption)
- [BLIB (Model From Huggingface)](https://www.kaggle.com/code/ahmedadel300/image-capination-blib)

## Credits

- [Ahmed Adel Mohammed](https://github.com/ahmeda335):
  - [LinkedIn](www.linkedin.com/in/ahmed-adel-b748aa23b)
  - [Contact](ahmedadel30320032003@gmail.com)
- [Ahmed Eltokhy](https://github.com/ahmdeltoky03):
  - [LinkedIn](https://www.linkedin.com/in/ahmd-eltokhey-8577b3275)
  - [Contact](ahmdeltoky4@gmail.com)
- [Amr Ghanem](https://github.com/AmrGhanem13):
  - [LinkedIn](https://www.linkedin.com/in/amr-ghanem-306b392b9)
  - [Contact](Amr_Ghanem07@yahoo.com)
- [Samir Mohammed](https://github.com/samir-m0hamed):
  - [LinkedIn](https://www.linkedin.com/in/samir-mohamed-2976bb252)
  - [Contact](samirmohamed122003@gmail.com)
- [Ahmed Elhefnawy](https://github.com/ahmedamrelhefnawy):
  - [LinkedIn](https://www.linkedin.com/in/ahmed-elhefnawy-258949243/)
  - [Contact](ahmedelhefnawy2003@hotmail.com)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
