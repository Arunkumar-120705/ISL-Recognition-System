# Real-Time Indian Sign Language Recognition

## Overview
This project presents a real-time Indian Sign Language (ISL) recognition system using hand landmarks and deep learning.

## Features
- Static gesture recognition (Alphabets, Numbers)
- Dynamic gesture recognition (Word-level)
- Transformer-based temporal modeling
- Real-time webcam inference

## Technologies Used
- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV

## Models
- Static Models:
  - Alphabets (126 features)
  - Numbers (63 features)
- Dynamic Model:
  - Transformer (30 × 126 sequence)

## How to Run

### Install dependencies
pip install -r requirements.txt

### Run dynamic test
python scripts/test_dynamic_video.py

### Run real-time
python scripts/realtime_dynamic_webcam.py

## Note
Dataset is not included due to size constraints.

## 📸 Results / Demo

### 🔤 Alphabet Recognition
![Alphabet 1](assets/screenshots/alphabet_test1.png)
![Alphabet 2](assets/screenshots/alphabet_test2.png)

### 🔢 Number Recognition
![Number 1](assets/screenshots/number_test1.png)
![Number 2](assets/screenshots/number_test2.png)

### 🎥 Dynamic Gesture Recognition
![Dynamic](assets/screenshots/dynamic_test.png)