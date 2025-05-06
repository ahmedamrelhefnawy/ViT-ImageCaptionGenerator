# Vision Transformer Image Caption Generator

This project implements an **Image Caption Generator** using a Vision Transformer (ViT) model. The application takes an image as input and generates a descriptive caption for it.

## Features

- **Vision Transformer (ViT)**: Utilizes a transformer-based architecture for image processing.
- **Streamlit Interface**: Provides an interactive web-based interface for uploading images and generating captions.
- **Preprocessing and Inference**: Includes modules for image preprocessing and caption generation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-caption-generator.git
   cd image-caption-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
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

## Example

1. Upload an image:
   ![Upload Example](https://via.placeholder.com/400x200?text=Upload+Image)

2. Generated caption:
   ```
   A group of people sitting at a table with food.
   ```

## Requirements

- Python 3.11+ (Recommended)
- TensorFlow
- Streamlit

Install additional dependencies listed in `requirements.txt`.

## Acknowledgments

- Vision Transformer (ViT) architecture inspired by [Google Research](https://github.com/google-research/vision_transformer).
- Streamlit for providing an easy-to-use interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
