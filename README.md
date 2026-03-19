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