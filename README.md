# CIFAR-10 Image Classification with ResNet50

A comprehensive deep learning project for classifying images from the CIFAR-10 dataset using transfer learning with ResNet50. This project includes extensive Exploratory Data Analysis (EDA) and a two-phase training approach.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Output Files](#output-files)
- [Key Features](#key-features)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🎯 Overview

This project demonstrates image classification on the CIFAR-10 dataset using transfer learning with a pre-trained ResNet50 model. The implementation includes:
- Comprehensive exploratory data analysis (EDA)
- Two-phase training strategy (feature extraction + fine-tuning)
- Visualization of training progress and results
- Model evaluation and performance metrics

## 📊 Dataset

**CIFAR-10** is a well-established computer vision dataset consisting of:
- **60,000 color images** (32x32 pixels)
- **10 classes** with 6,000 images per class
- **50,000 training images** and **10,000 test images**

### Classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

**Note:** For this project, we use **10,000 training samples** (1,000 per class) to balance training time and model performance. The full test set of 10,000 images is used for evaluation.

## 📁 Project Structure
```
cifar10-project/
│
├── cifar10_resnet50.py          # Main project code
├── README.md                     # This file
│
├── outputs/                      # Generated files
│   ├── eda_1_class_samples.png
│   ├── eda_2_class_distribution.png
│   ├── eda_3_pixel_intensity.png
│   ├── eda_4_average_images.png
│   ├── eda_5_brightness_analysis.png
│   ├── eda_6_random_grid.png
│   ├── training_history.png
│   └── cifar10_resnet50_model.h5
```

## 🛠️ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/cifar10-project.git
cd cifar10-project
```

2. **Install dependencies:**
```bash
pip install tensorflow numpy matplotlib seaborn
```

Or using conda:
```bash
conda install tensorflow numpy matplotlib seaborn
```

3. **Verify installation:**
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 🚀 Usage

### Running in Jupyter Notebook

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Create a new notebook and copy the code from `cifar10_resnet50.py`

3. Run all cells sequentially

### Running as Python Script
```bash
python cifar10_resnet50.py
```

### Expected Runtime
- **EDA Phase:** ~2-3 minutes
- **Phase 1 Training (10 epochs):** ~15-20 minutes
- **Phase 2 Training (10 epochs):** ~20-25 minutes
- **Total:** ~40-50 minutes (depending on hardware)

## 🏗️ Model Architecture

### Base Model: ResNet50
- Pre-trained on ImageNet
- Input shape: (32, 32, 3)
- All layers frozen during Phase 1

### Custom Classification Head
```
ResNet50 Base (frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(10, activation='softmax')
```

**Total Parameters:** ~23.8 million
- ResNet50 base: ~23.6 million
- Custom head: ~200k

## 🎓 Training Strategy

### Phase 1: Feature Extraction (10 epochs)
- **Objective:** Train only the custom classification head
- **Base model:** Frozen (weights not updated)
- **Learning rate:** 0.001
- **Batch size:** 32
- **Validation split:** 20%

### Phase 2: Fine-Tuning (10 epochs)
- **Objective:** Fine-tune the entire network
- **Base model:** Unfrozen (all layers trainable)
- **Learning rate:** 0.0001 (reduced to prevent overfitting)
- **Batch size:** 32
- **Validation split:** 20%

### Why Two Phases?
1. **Phase 1** allows the custom head to learn appropriate feature representations without disrupting pre-trained weights
2. **Phase 2** fine-tunes the entire network for optimal performance on CIFAR-10

## 📈 Results

### Expected Performance
- **Training Accuracy:** ~75-85%
- **Validation Accuracy:** ~70-80%
- **Test Accuracy:** ~70-80%

*Note: Results may vary based on random initialization and hardware*

### Training Curves
The project generates a visualization showing:
- Training vs. Validation Accuracy
- Training vs. Validation Loss
- Clear marker indicating when the base model was unfrozen

## 📤 Output Files

The project generates **8 files**:

### Visualizations (7 PNG files)
1. **eda_1_class_samples.png** - Sample images from each class
2. **eda_2_class_distribution.png** - Bar charts showing class balance
3. **eda_3_pixel_intensity.png** - RGB channel intensity histograms
4. **eda_4_average_images.png** - Average representation of each class
5. **eda_5_brightness_analysis.png** - Brightness distribution by class
6. **eda_6_random_grid.png** - Random sample grid of 60 images
7. **training_history.png** - Accuracy and loss curves over 20 epochs

### Model (1 H5 file)
8. **cifar10_resnet50_model.h5** - Trained model (can be loaded for inference)

## ✨ Key Features

- ✅ **Comprehensive EDA:** 6 different visualizations to understand the dataset
- ✅ **Transfer Learning:** Leverages pre-trained ResNet50 for better performance
- ✅ **Two-Phase Training:** Systematic approach to training deep networks
- ✅ **Balanced Dataset:** Equal representation of all classes
- ✅ **Reproducible Results:** Clear code structure and documentation
- ✅ **Visual Feedback:** Training progress and model performance visualizations
- ✅ **Model Persistence:** Trained model saved for future use

## 🔧 Customization

### Adjust Training Sample Size
Change the value of `n` in the code:
```python
n = 10000  # Use 10,000 samples (default)
n = 5000   # Faster training, potentially lower accuracy
n = 20000  # Better accuracy, longer training time
n = 50000  # Full dataset, maximum accuracy, very slow
```

### Modify Model Architecture
Add or remove layers in the custom head:
```python
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),  # Add more neurons
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),  # Add extra layer
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
```

### Adjust Training Parameters
```python
# Learning rates
learning_rate_phase1 = 0.001  # Phase 1
learning_rate_phase2 = 0.0001  # Phase 2

# Number of epochs
epochs_phase1 = 10
epochs_phase2 = 10

# Batch size
batch_size = 32  # Increase for faster training (if memory allows)
```

## 🐛 Troubleshooting

### Issue: "Cannot import TensorFlow"
```bash
pip install --upgrade tensorflow
```

### Issue: "Out of Memory" error
- Reduce batch size: `batch_size = 16`
- Reduce training samples: `n = 5000`

### Issue: Plots not showing in Jupyter
Add this at the top of your notebook:
```python
%matplotlib inline
```

### Issue: Slow training
- Ensure you're using GPU if available
- Reduce number of training samples
- Reduce number of epochs

### Issue: Low accuracy
- Increase training samples: `n = 20000` or more
- Train for more epochs
- Adjust learning rates
- Add data augmentation

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- CIFAR-10 dataset creators: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- ResNet architecture: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
- TensorFlow and Keras teams for excellent deep learning frameworks



