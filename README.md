# Hand Gesture Recognition

# Project Overview

Hand Gesture Recognition is a computer vision project that detects a user's hand through a webcam and recognizes basic hand gestures in real time.

The project uses OpenCV for webcam input and MediaPipe Hand Landmarker for detecting hand landmarks. The detected landmarks are used to classify simple hand gestures and map them to different actions.

# Features

- Real-time webcam input
- Hand detection using MediaPipe
- Detection of hand landmarks
- Recognition of basic hand gestures
- Gesture-to-action mapping
- Real-time display of gesture and action
- Simple and easy-to-use Python application

## Technologies Used

- Python
- OpenCV
- MediaPipe

# Gestures and Actions

| Gesture | Action |
|---------|--------|
| Fist | Pause |
| Open Palm | Play |
| One Finger | Volume Up |
| Two Fingers | Volume Down |
| Three Fingers | Next |

# How It Works

The application follows these steps:

1. The webcam captures the video.
2. OpenCV processes the camera frames.
3. MediaPipe detects the user's hand.
4. MediaPipe provides hand landmark points.
5. The landmark positions are analyzed to identify the gesture.
6. The recognized gesture is mapped to an action.
7. The gesture and corresponding action are displayed on the screen.

# Project Structure

Hand-Gesture-Recognition/
│
├── models/
│   └── hand_landmarker.task
│
├── venv/
│
├── main.py
│
└── README.md
