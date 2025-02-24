# DeepFake Detectron

DeepFake Detectron is a deep learning-based system designed to detect deepfake videos and images using state-of-the-art convolutional neural networks. This project leverages **VGG16**, **ResNet50**, and **InceptionV3** models to achieve high accuracy in identifying manipulated media.

## Features

- **Multi-Model Approach**: Utilizes VGG16, ResNet50, and InceptionV3 for robust detection.
- **Pretrained Weights**: Uses pretrained models on ImageNet and fine-tunes them for deepfake detection.
- **High Accuracy**: Combines the strengths of multiple architectures to enhance detection performance.
- **Scalability**: Can be deployed in real-time applications for video analysis.

## Technologies Used

- **Deep Learning Framework**: TensorFlow & PyTorch
- **Models**: VGG16, ResNet50, InceptionV3
- **Dataset**: DeepFake detection datasets
- **Backend**: Flask for API deployment

## Getting Started

### Prerequisites
Ensure you have the following installed:

- Python 3.8+
- TensorFlow & PyTorch
- OpenCV
- NumPy & Pandas
- Flask 

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/DeepFake-Detectron.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd DeepFake-Detectron
   ```

## Usage

### Running API Server
To serve the model via API:
```bash
python app.py
```
The API will be available at `http://localhost:8000`.
---

Thank you for using DeepFake Detectron! 🚀
