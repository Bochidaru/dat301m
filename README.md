# Image Captioning Project

A comprehensive image captioning project that generates Vietnamese captions for images using multiple deep learning architectures. This project implements Vision Transformer (ViT) + T5, MobileNet + RNN, and ResNet + LSTM models for image captioning.

## Features

- **Multiple Model Architectures**:
  - Vision Transformer (ViT) + T5 (Vietnamese)
  - MobileNet + RNN
  - ResNet101 + LSTM

- **Vietnamese Language Support**: Uses VietAI's ViT5-base model for generating Vietnamese captions

- **Dataset Support**:
  - COCO dataset (with Vietnamese captions)
  - Flickr30k dataset
  - KTViC dataset

- **REST API**: FastAPI service for real-time image captioning with text-to-speech (TTS) support

- **Data Processing Tools**: Preprocessing and visualization notebooks for dataset analysis

## Project Structure

```
dat301m/
├── data_preprocessing/          # Data preprocessing notebooks
│   ├── merge_data.ipynb
│   ├── Create_training_dataset.ipynb
│   ├── flickr30k-vinai-translate.ipynb
│   └── json_to_csv_converter.py
├── data_visualization/          # Data visualization tools
│   ├── flickr30k_word_frequency_analysis.ipynb
│   └── visualize_images.ipynb
├── vit/                         # Vision Transformer + T5 implementation
│   ├── config.py                # Configuration settings
│   ├── data.py                  # Data loading utilities
│   ├── modeling_vitt5.py        # ViT-T5 model architecture
│   ├── train.py                 # Training script
│   ├── train_utils.py           # Training utilities
│   └── eval_utils.py            # Evaluation utilities
├── mobilenet_rnn/               # MobileNet + RNN implementation
│   ├── train.ipynb
│   ├── resplit.ipynb
│   └── vocab.txt
├── resnet_lstm/                 # ResNet + LSTM implementation
│   └── resnet101-lstm.ipynb
└── docker/                      # Docker deployment
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        ├── app.py               # FastAPI application
        └── model.py             # Model inference code
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dat301m
```

2. Install dependencies:

For ViT-T5 training:
```bash
cd vit
pip install tensorflow transformers pillow pandas scikit-learn pycocotools
```

For Docker deployment:
```bash
cd docker
pip install -r requirements.txt
```

## Usage

### Training ViT-T5 Model

The main training script is located in `vit/train.py`. You can configure training parameters through command-line arguments or by modifying `vit/config.py`.

```bash
cd vit
python train.py \
    --dataset coco-vi-human \
    --batch_size 32 \
    --epochs 10 \
    --acc_steps 8 \
    --lr_proj 3e-4 \
    --lr_t5 5e-5 \
    --mixed_precision
```

**Available datasets:**
- `coco-vi-human`: COCO dataset with Vietnamese human-annotated captions
- `coco-vi-human2`: Alternative COCO Vietnamese dataset
- `ktvic`: KTViC dataset

**Configuration Options:**
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--acc_steps`: Gradient accumulation steps (default: 8)
- `--lr_proj`: Learning rate for projection layer (default: 3e-4)
- `--lr_t5`: Learning rate for T5 decoder (default: 5e-5)
- `--mixed_precision`: Enable mixed precision training

### Running the API Server

Start the FastAPI server for image captioning inference:

```bash
cd docker/app
python app.py
```

Or using Docker:

```bash
cd docker
docker build -t image-captioning-api .
docker run -p 8000:8000 image-captioning-api
```

**API Endpoints:**

1. **Generate Caption** (POST `/generate_caption`):
   - Upload an image file
   - Returns: Vietnamese caption and base64-encoded audio (text-to-speech)

   Example:
   ```bash
   curl -X POST "http://localhost:8000/generate_caption" \
        -F "file=@image.jpg" \
        -F "voice=vi-VN-HoaiMyNeural" \
        -F "rate=+30%"
   ```

2. **Health Check** (GET `/`):
   - Returns API status

### Data Preprocessing

Use the notebooks in `data_preprocessing/` to:
- Merge and process datasets
- Convert JSON annotations to CSV format
- Translate captions using Vinai translation models
- Create training datasets

### Visualization

Explore the notebooks in `data_visualization/` to:
- Analyze word frequency in captions
- Visualize images with their captions
- Generate statistics about the dataset

## Configuration

The ViT-T5 model configuration is managed in `vit/config.py`. Key settings include:

- **Model**: Vision Transformer (ViT-base-patch16-224) + T5-base (Vietnamese)
- **Image Size**: 224x224
- **Max Caption Length**: 35 tokens
- **Batch Size**: 32 (with gradient accumulation)
- **Learning Rates**: Separate learning rates for projection layer and T5 decoder
- **Mixed Precision**: Enabled by default for faster training

## Model Architecture

### ViT-T5 Model

1. **Vision Encoder**: Vision Transformer (ViT) extracts image features
2. **Projection Layer**: Maps ViT features to T5 embedding space
3. **Text Decoder**: T5 decoder generates Vietnamese captions autoregressively

The model uses transfer learning:
- ViT encoder: Pre-trained on ImageNet (frozen during initial training)
- T5 decoder: Pre-trained VietAI/vit5-base (fine-tuned)

## Experiments

The project includes implementations of multiple architectures:

1. **ViT-T5** (`vit/`): State-of-the-art transformer-based approach
2. **MobileNet-RNN** (`mobilenet_rnn/`): Lightweight CNN with RNN decoder
3. **ResNet-LSTM** (`resnet_lstm/`): Deep CNN with LSTM decoder

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- VietAI for the Vietnamese T5 model (vit5-base)
- Hugging Face Transformers library
- COCO dataset team
- Flickr30k dataset creators

## Contact

For questions or issues, please open an issue on the repository.

---

**Note**: Make sure to set up your dataset paths correctly in the configuration files before training or inference.

