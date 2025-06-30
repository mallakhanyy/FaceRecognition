# Real-Time Face Recognition

A deep learning–powered real-time face recognition system built from scratch with a clean Tkinter GUI and fully modular design.

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project includes:
- Face Detection using **MTCNN**
- Face Embedding using **FaceNet**
- Identity Classification using **SVM**
- Desktop GUI built with **Tkinter**
- Reusable saved model and dataset for fast integration

---

## Phase 1: Model Development & Training (`firstfacerecognitionmodel.py`)

### 1. Face Detection with MTCNN
Detects and extracts faces from images — robust to lighting, pose, and background variation.

### 2. Face Embedding with FaceNet
Each face is converted to a 512D vector representing unique identity features.

### 3. Embedding Normalization
L2 normalization applied for consistent vector scale.

### 4. SVM Classification
A linear SVM model with probability support is trained on normalized embeddings.

### 5. Model Evaluation
Includes:
- Training accuracy
- Testing accuracy
- Classification report
- Confusion matrix

### 6. Model Saving for Reuse
- `face_recognition.pkl` → Trained SVM + label encoder  
- `faces_dataset.npz` → All face embeddings + labels

These files are reusable in any future app.

---

## Phase 2: Real-Time Desktop App (`app.py`)

### Features
- Captures frames via webcam
- Detects face with MTCNN
- Generates FaceNet embeddings
- Predicts identity using trained SVM
- Displays results in GUI window

### Box Colors
- Green: Recognized face
- Blue: Unrecognized face

### User Interface
- Tkinter GUI with modern themed colors
- Dynamic label shows prediction in real-time
- Custom icon and exit button

---

## Color Theme

| Component      | Color     |
|----------------|-----------|
| Background     | `#F6DED8` |
| Text           | `#D2665A` |
| Title          | `#B82132` |
| Button BG      | `#F2B28C` |
| Button Text    | `#ffffff` |
| Unrecognized   | Blue (RGB)

Color palette link: [Click here](https://colorhunt.co/palette/b82132d2665af2b28cf6ded8)

---

## Smart Architecture

- Loads trained model and label encoder only
- Face embeddings are computed live
- Recognition starts immediately
- Model supports structured dataset of multiple people

---

## Highlights

- Full AI pipeline from detection to prediction
- Real-time face recognition
- Saved model/data for reuse
- GUI for clean user experience

---

## Installation

```bash
pip install -r requirements.txt
```


## Author

**Malak Hany**  
[LinkedIn](https://www.linkedin.com/in/mallakhanyy)  
mallak.hanyy@gmail.com
