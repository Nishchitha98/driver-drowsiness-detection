#Driver Drowsiness Detection System 
Project: Driver Drowsiness Detection System
Author: NISHCHITHA N R
Semester / Course: 5th Semester — Computer Science & Engineering
Institution: MAHARAJA INSTITUE OF TECHNOLOGY MYSORE
Date: November 18TH,2025

Table of Contents
Project Overview
Motivation
Features
Architecture & Methodology
Repository Structure
Requirements / Dependencies
Setup & Installation
Run Instructions
Usage / Demo
Dataset / Assets
Limitations
Future Work

Project Overview
A real-time driver drowsiness detection system using MediaPipe Face Mesh and Eye Aspect Ratio (EAR). The system streams webcam video to a Flask web UI, calculates EAR and MAR, detects drowsiness or yawning, and triggers an alarm.

Motivation
Road accidents caused by driver fatigue are a serious concern worldwide, resulting in loss of life and property every year. Existing solutions, such as wearable sensors or advanced vehicle mounted systems, are often expensive, intrusive, and not easily accessible for everyday drivers. This project was chosen to develop a simple, cost-effective, and non-intrusive solution using AI and computer vision techniques. It also provides an opportunity to gain hands-on experience in machine learning, computer vision, and real-time system development.

Features
Real-time face, eye and mouth detection using MediaPipe Face Mesh
Eye Aspect Ratio (EAR) for eye closure detection
MAR (mouth aspect ratio) for yawn detection (optional)
Head-down detection (simple nose Y coordinate heuristic)
Visual alert on UI + audible alarm (assets/alarm.wav)
Flask-based web UI with Start / Stop camera controls
Lightweight, runs on a laptop with webcam
Architecture & Methodology
Capture frames from webcam.
Use MediaPipe Face Mesh to extract facial landmarks.
Compute EAR (and MAR) from landmark coordinates.
If EAR < threshold for N consecutive frames or head-down condition → mark as Drowsy.
Play alarm sound and display alert on the browser stream served by Flask.

Repository Structure
DRIVER_DROWSINESS_DETECTION/ ├── assets/

│

└── alarm.wav 

├── src/ 

│

├── app.py # Flask app (starts server, routes)

├── drowsiness_ear.py # MediaPipe detection + process_frame() 

├── templates/ 

│

└── index.html

├── README.md

└── requirements.txt

Requirements / Dependencies
Python 3.10 (recommended) — the project was tested on Python 3.10.10
Packages (install via pip): listed in requirements.txt
mediapipe
opencv-python
numpy
pygame
flask

Setup & Installation (Windows)
Clone the repo (or copy files).
Create a virtual environment and activate:
python -m venv venv
venv\Scripts\activate
Install dependencies: pip install -r requirements.txt
Verify assets/alarm.wav exists.

Run Instructions

Run the Flask app: venv\Scripts\activate python src\app.py

Open browser: http://127.0.0.1:5000 Click Start Camera to begin streaming and detection.

Usage / Demo

Start camera → video appears in UI. 
Close eyes for more than threshold frames → Drowsiness Alert shows on video and alarm sounds. 
Press Stop Camera → stops stream and releases camera

Dataset / Assets

assets/alarm.wav — alarm sound used for alerts data/ — optional folder where captured eye crops are stored if you use dataset generation mode

Limitations

Works in normal lighting; may degrade under low light EAR thresholds may need tuning per person / camera Not a full substitute for production-grade driver monitoring (no IR camera, no multi-camera support)

Future Work

Add head pose estimation using solvePnP for more accurate nod detection 
Add database/logging and SMS/E-mail notifications Mobile/web deployment and model optimization
