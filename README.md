# Ink-Trace

A deep learning-based Optical Character Recognition (OCR) system built entirely in **PyTorch** for recognizing handwritten text using a **Convolutional Recurrent Neural Network (CRNN)** with **Connectionist Temporal Classification (CTC)**.

Ink-Trace learns to convert grayscale handwritten text images into character sequences without requiring explicit character segmentation. The project is trained on the IAM Handwriting Database and includes a complete training, evaluation, inference, and web-based testing pipeline.

---

## Features

* **Custom CRNN Architecture** combining a CNN feature extractor with a Bidirectional LSTM for sequence modeling.
* **CTC Loss** for end-to-end handwritten text recognition without character-level annotations.
* **Aspect-Ratio Preserving Preprocessing** with fixed-height resizing while maintaining handwriting geometry.
* **Automatic Character Vocabulary Generation** from the training dataset.
* **Mixed Precision (AMP) Training** for faster GPU training and reduced memory usage.
* **Character Error Rate (CER) Evaluation** for quantitative model assessment.
* **Interactive Streamlit Interface** for testing OCR predictions on custom handwritten images.

---

# Model Architecture

The OCR model consists of three stages:

### 1. Feature Extraction (CNN)

A VGG-inspired convolutional backbone extracts visual features from the input image.

* 4 Convolution layers
* Batch Normalization
* ReLU activations
* Max Pooling

The CNN converts the handwriting image into a sequence of feature vectors.

---

### 2. Sequence Modeling (BiLSTM)

The extracted feature sequence is processed using a Bidirectional LSTM.

* Hidden Size: **512**
* Multiple recurrent layers
* Bidirectional sequence learning
* Dropout regularization

The BiLSTM learns contextual dependencies across the handwritten text.

---

### 3. Transcription (CTC)

A fully-connected layer predicts character probabilities for every timestep.

Training uses:

* Connectionist Temporal Classification (CTCLoss)

Inference uses:

* Greedy CTC decoding

---

# Project Structure

```text
Ink-Trace/
│
├── checkpoints/
│   └── *.pth
│
├── data/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
│
├── src/
│   ├── data_loader.py
│   └── model.py
│
├── app.py
├── train.py
├── predict.py
├── test.py
├── vocab.py
├── char_map.json
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/SID-SAN/Ink-Trace.git

cd Ink-Trace
```

Create a virtual environment:

```bash
conda create -n torch python=3.10

conda activate torch
```

Install dependencies:

```bash
pip install requirements.txt
```

---

# Dataset

The project expects IAM-style parquet files containing:

* image.bytes
* text

Example directory:

```text
data/
├── train.parquet
├── validation.parquet
└── test.parquet
```

---

# Usage

## 1. Build Vocabulary

Generate the character mapping:

```bash
python vocab.py
```

---

## 2. Train

```bash
python train.py
```

Features:

* CUDA support
* Automatic Mixed Precision (AMP)
* Gradient clipping
* Learning-rate scheduler
* Checkpoint saving

---

## 3. Evaluate

Evaluate Character Error Rate (CER):

```bash
python test.py
```

Metrics reported:

* Character Error Rate (CER)
* Exact Match Accuracy

---

## 4. Predict

Run OCR on a single image:

```bash
python predict.py
```

---

## 5. Launch the Web Interface

Start the Streamlit application:

```bash
streamlit run app.py
```

The interface allows you to:

* Upload handwritten images
* Run OCR inference
* View predicted text
* Inspect decoded character indices for debugging

---

# Current Results

The current implementation successfully performs end-to-end handwritten text recognition and demonstrates:

* CNN feature extraction
* Bidirectional sequence modeling
* CTC-based transcription
* Interactive OCR inference through Streamlit

Performance is strongest on short handwritten words and phrases. Recognition accuracy decreases on long IAM text lines due to the computational trade-offs involved in processing very wide handwritten sequences.

---

# Future Improvements

* Beam Search decoding
* Dynamic-width batching to eliminate fixed-width truncation
* Stronger data augmentation
* Transformer/Conformer-based sequence models
* Language-model-assisted decoding
* Writer-specific fine-tuning
* ONNX / TorchScript export for deployment

---

# Technologies Used

* Python
* PyTorch
* TorchVision
* Streamlit
* Pandas
* Pillow
* CUDA
* Automatic Mixed Precision (AMP)

---
