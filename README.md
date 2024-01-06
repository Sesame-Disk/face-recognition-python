# Python Face Recognition Tutorial

A comprehensive guide for beginners to learn face recognition using Python.

## Prerequisites
- Python 3+
- IDE (VS Code, PyCharm, Google Colab, etc.)
- Clean dataset of images
- Basic coding knowledge

## Libraries
- `face_recognition`
- `NumPy`
- `OpenCV`
- `os`
- `DateTime`
- `deepface`
- `matplotlib`
- `pprint`


## Key Features

* Handles face detection, alignment, normalization, representation, and verification under the hood.
*Supports various backend detectors (OpenCV, SSD, Dlib, MTCNN, RetinaFace, MediaPipe) for flexibility.
* Offers model selection, including VGG-Face and the highly accurate Facenet-512.
# Provides functions for:
* Finding faces matching a given image within a database.
* Extracting face encodings for storage and later comparison.


## Installation
Ensure CMake and Dlib are installed. Then:



## Getting Started

Install dependencies:
```
pip install deepface
pip install face_recognition
pip install deepface
pip install opencv
pip install matplotlib
```

Clone the repository:
```
git clone https://github.com/Sesame-Disk/face-recognition-python.git
```


## Steps

1. **Import Libraries**
   Import necessary libraries like `cv2`, `numpy`, `face_recognition`, `deepface`, etc.

2. **Setup Image Path**
   Declare a variable for the image folder path and create lists for image names and encodings.

3. **Find Face Encodings**
   Use `face_recognition` to find and store face encodings from images.

4. **Video Camera Capture**
   Use OpenCV to capture live video and detect faces.

5. **Face Locations and Distances**
   Implement face location detection and calculate face distances using NumPy.

6. **Compare Face Encodings**
   Compare faces using thresholds to identify matches.

7. **Face Match Percentage**
   (Optional) Print face match percentage based on comparison results.

8. **Exit Webcam**
   Properly release webcam resources after usage.

## DeepFace Integration
- Utilize deepface for advanced face recognition and analysis.
- Supports various backends and models like VGG-Face, Facenet-512.
- Analyze attributes like sentiment, age, sex, and ethnicity.

## Face Verification with DeepFace
- Match images for similarity and identify if they belong to the same person.
- Supports multiple models and backends for robust verification.

## Dockerizing the Face Recognition Project

Dockerizing your Python face recognition project can simplify deployment and ensure consistency across various environments. Follow these steps to containerize your application:

### 1. Create a Dockerfile
Create a `Dockerfile` in your project root with the necessary commands to build the Docker image. This includes setting up the Python environment, installing dependencies, and specifying the entry point for your application.

### 2. Build the Docker Image
Run `docker build -t python-face-recognition .` to build your Docker image.

### 3. Run the Docker Container
Execute `docker run -p 5000:5000 python-face-recognition` to start the container. Adjust the port settings as needed for your application.

### 4. Docker Compose (Optional)
For more complex setups, use Docker Compose to manage multi-container Docker applications.

By Dockerizing your face recognition project, you can streamline the deployment process and ensure a uniform environment for development, testing, and production.

## Conclusion
This tutorial provides a solid foundation for understanding and implementing face recognition in Python, covering essential concepts and practical implementation steps using OpenCV, deepface, and other key libraries.

## Additional Resources
- Code Files: [GitHub Repository](https://github.com/Sesame-Disk/face-recognition-python)
- Tutorial Article: [Face Recognition in Python, by Syed Umar Bukhari](https://sesamedisk.com/face-recognition-in-python-a-beginners-guide/)
