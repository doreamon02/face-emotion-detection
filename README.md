# Face Emotion Detection

A Deep Learning and Computer Vision project that detects human emotions from facial expressions in real time. The system leverages DeepFace and OpenCV to analyze facial expressions and classify emotions such as Happy, Sad, Angry, Fear, Surprise, Neutral, and Disgust.

## Overview

This project uses DeepFace for emotion analysis and facial recognition capabilities. It supports real-time emotion detection and maintains a facial database for identifying known individuals.

## Features

* Real-time face and emotion detection
* Emotion classification from images and webcam streams
* Face database creation and management
* Recognition of known individuals
* Integration with DeepFace and OpenCV
* Accurate emotion prediction

## Technologies Used

* Python
* DeepFace
* TensorFlow
* OpenCV
* NumPy
* Pandas

## Project Structure

```text
Face-Emotion-Detection/
│
├── app_emotion.py              # Main application for emotion detection
├── create_deepface_db.py       # Creates and manages the DeepFace database
├── emotion_detection_db.py     # Emotion detection with database integration
├── test_deepface.py            # Script for testing DeepFace functionality
├── requirements.txt            # Project dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/doreamon02/Face-Emotion-Detection.git
cd Face-Emotion-Detection
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python app_emotion.py
```

Create the face database:

```bash
python create_deepface_db.py
```

Run emotion detection with database support:

```bash
python emotion_detection_db.py
```

Test DeepFace functionality:

```bash
python test_deepface.py
```

## Applications

* Human-Computer Interaction
* Smart Surveillance Systems
* Emotion-Aware AI Systems
* Customer Sentiment Analysis
* Personalized User Experiences

## Author

**Mohammad Armaan**

GitHub: https://github.com/doreamon02

LinkedIn: https://www.linkedin.com/in/armaan-fulara-80333b1b6

## License

This project is licensed under the MIT License.

