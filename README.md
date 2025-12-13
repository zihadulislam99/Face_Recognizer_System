[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CONTRIBUTING.md)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub Stars](https://img.shields.io/github/stars/zihadulislam99/REPO_NAME?style=social)](https://github.com/zihadulislam99/REPO_NAME/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/zihadulislam99/REPO_NAME?style=social)](https://github.com/zihadulislam99/REPO_NAME/network/members)

# Face Recognizer System

A **real-time face recognition system** built with Python and OpenCV that detects and recognizes human faces using a webcam. It uses Haar Cascade classifiers for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for recognizing registered users. Unknown faces are marked as “Unknown,” and confidence scores are displayed for each prediction. The system is modular, lightweight, and suitable for applications like attendance tracking, access control, and personal computer vision projects.

---

## **Features**

* **Real-Time Face Detection:** Detects faces from webcam video feed in real time.
* **Face Recognition:** Identifies registered users using a trained LBPH model.
* **Unknown Face Detection:** Marks unregistered or unrecognized faces as "Unknown."
* **Confidence Score:** Displays a confidence score for each recognition.
* **Custom Dataset Support:** Allows creation of user-specific datasets for accurate recognition.
* **Modular Design:** Separate scripts for dataset creation, training, and recognition.
* **Lightweight and Fast:** Efficient for small-to-medium scale face recognition tasks.

---

## **Technology Stack**

* **Programming Language:** Python
* **Libraries:** OpenCV (with contrib), NumPy, Pillow (PIL)
* **Face Detection:** Haar Cascade Classifier
* **Face Recognition Algorithm:** LBPH (Local Binary Patterns Histogram)

---

## **Project Structure**

```
Face-Recognizer-System/
│
├── data/                       # Folder to store face images
│   └── user.ID.imgID.jpg       # Example: user.1.1.jpg
│
├── dataset_creator.py          # Script to capture images via webcam
├── classification.py           # Script to train the LBPH classifier
├── face_recognizer.py          # Real-time recognition script
├── classifier.xml              # Trained LBPH model (after training)
├── haarcascade_frontalface_default.xml  # Face detection Haarcascade
└── README.md                   # Project documentation
```

---

## **Setup Instructions**

### 1. Install Dependencies

```bash
pip install opencv-contrib-python numpy pillow
```

> **Note:** Use `opencv-contrib-python` to access the `cv2.face` module.

---

### 2. Create Dataset

Run `dataset_creator.py` to capture multiple images of each user via webcam.

* Images are stored in the `data/` folder.
* Filename format: `user.ID.imgID.jpg` (e.g., `user.1.1.jpg`).
* Capture 20–50 images per user from different angles for better accuracy.

---

### 3. Train Classifier

Run `classification.py` to train the LBPH face recognizer.

* The script generates `classifier.xml`.
* This model is used for recognizing faces in real time.

---

### 4. Run Face Recognition

Run `face_recognizer.py` to start real-time recognition.

* Webcam feed opens.
* Recognized faces display names and confidence scores.
* Unknown faces are labeled as `Unknown`.
* Press `q` to exit.

---

## **Usage Example**

```bash
python dataset_creator.py      # Capture images for each user
python classification.py       # Train the LBPH model
python face_recognizer.py      # Start real-time recognition
```

---

## **Tips for Accuracy**

* Capture multiple images per user with different angles and lighting conditions.
* Avoid occlusions like masks or sunglasses when capturing dataset.
* Ensure faces are clearly visible in the webcam frame.

---

## **Applications**

* Attendance tracking in schools, colleges, and offices.
* Security and access control systems.
* Personal or research projects in computer vision.
* Any application requiring automated user identification.

---

## **License**

This project is **open-source** and free to use for personal, educational, or research purposes.
