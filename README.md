# GFPGAN Image Enhancement

This project uses the GFPGAN (Generative Facial Prior-Generative Adversarial Network) model for image enhancement. The application allows users to upload images, process them using GFPGAN, and view both the original and enhanced images.

## Features

- Upload images for enhancement.
- Process images with the GFPGAN model.
- View the original and enhanced images side by side.

## Installation

1. **Clone the repository:**

   \\\ash
   git clone https://github.com/your-repository-url.git
   cd your-repository-folder
   \\\

2. **Create and activate a virtual environment:**

   \\\ash
   python -m venv venv
   venv\\Scripts\\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   \\\

3. **Install dependencies:**

   \\\ash
   pip install -r requirements.txt
   \\\

4. **Download pre-trained models:**

   Models are dynamically downloaded when first used. They will be saved in the \experiments/pretrained_models\ directory.

## Usage

1. **Run the Flask application:**

   \\\ash
   python app.py
   \\\

2. **Open your web browser and go to:**

   \\\
   http://127.0.0.1:5000
   \\\

3. **Upload an image:**

   - Navigate to the homepage.
   - Use the upload form to select and submit an image.

4. **View results:**

   - After processing, you will be redirected to a page showing the original and enhanced images.

## Image Comparisons

The following images illustrate the input and corresponding output results:

### Inputs

![Original Image 1](static/uploads/"10 per.jpg")
![Original Image 2](static/uploads/"40 per.jpg")

### Outputs

![Enhanced Image 1](static/results/10 per.jpg)
![Enhanced Image 2](static/results/40 per.jpg)

## Troubleshooting

- **Model Download Issues:** Ensure you have an active internet connection. Models are downloaded automatically if they are not found in the \experiments/pretrained_models\ directory.
- **File Corruption:** If you encounter issues with corrupted model files, delete the existing models in \experiments/pretrained_models\ and restart the application to re-download them.
- **Torchvision.transformers.functional_tensor Error:** Incase the module not found error arrives for torchvision then Open /venv/lib/python3.10/site-packages/basicsr/data/degradations.py and on line 8, simply change:

from torchvision.transforms.functional_tensor import rgb_to_grayscale
to:

from torchvision.transforms.functional import rgb_to_grayscale

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [GFPGAN](https://github.com/TencentARC/GFPGAN) for the image enhancement model.
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) for the background upscaling model.

