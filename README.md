Computer Vision

A hands-on learning repository covering classic image processing, feature/object detection, and CNN architectures — implemented from first principles (raw NumPy) as well as with OpenCV, TensorFlow/Keras, and Ultralytics YOLOv8.

📁 Repository Structure
Computer_Vision/
├── Conv.ipynb                  # Convolution, padding & stride from scratch (NumPy)
├── edgedetection.ipynb         # Sobel edge detection from scratch (NumPy)
├── imageProcessing.ipynb       # OpenCV basics: grayscale, blur, Canny edges, thresholding
├── countours.ipynb             # Contour detection with OpenCV
├── FeatureDetection.ipynb      # Harris corners, FAST, and ORB feature detection (webcam)
├── faceDetection.ipynb         # Haar Cascade face detection (image + live webcam)
├── objectDetection.ipynb       # YOLOv8 object detection (image + live webcam)
├── VideoProcessing.ipynb       # OpenCV webcam fundamentals (capture, grayscale, flip, FPS, save frame)
├── basic_cnn.ipynb             # CNN from scratch: Dogs vs Cats binary classifier
├── LeNetArchitecture.ipynb     # LeNet-5 implementation on MNIST
├── AlexNet.ipynb               # AlexNet-style CNN on CIFAR-10
├── yolov8n.pt                  # Pretrained YOLOv8 nano weights (used by objectDetection.ipynb)
└── apple.jpg, messi.jpg, humanface.jpg, images.png, messi_gray.png   # Sample images used across notebooks
🧩 1. Image Processing & Feature Extraction

Fundamentals of computer vision, several implemented manually to understand the underlying math before relying on library calls.

Notebook	What it does
Conv.ipynb	Implements 2D convolution, zero-padding, and strided convolution from scratch with NumPy (no OpenCV/TensorFlow) on a small test matrix and a custom kernel.
edgedetection.ipynb	Implements the Sobel operator manually by sliding a vertical kernel over a grayscale image, comparing the result to the original image.
imageProcessing.ipynb	OpenCV basics — grayscale conversion, Gaussian blur, Canny edge detection, and binary thresholding.
countours.ipynb	Detects and draws contours on a thresholded image using cv2.findContours / cv2.drawContours.
FeatureDetection.ipynb	Real-time keypoint/corner detection from a webcam feed using Harris Corner Detection, the FAST algorithm, and ORB descriptors.
🕵️ 2. Detection Models
Notebook	What it does
faceDetection.ipynb	Face detection with OpenCV's Haar Cascade classifier (haarcascade_frontalface_default.xml) — first on a static image, then live on a webcam feed.
objectDetection.ipynb	General-purpose object detection using a pretrained YOLOv8n model (yolov8n.pt, via Ultralytics) — run on a still image and on a live webcam feed with real-time annotated output.
🎥 3. Video Processing
Notebook	What it does
VideoProcessing.ipynb	Core OpenCV video/webcam operations: capturing live video, printing frame shape, converting to grayscale, mirroring the feed, reading FPS, and saving a captured frame to disk.
🧠 4. CNN Architectures
Notebook	Task	Dataset	Test Result
basic_cnn.ipynb	Binary image classification (cat vs. dog) with a custom 4-block CNN	Dogs vs. Cats (Kaggle)	85.6% accuracy, loss 0.457
LeNetArchitecture.ipynb	Digit classification using a LeNet-5 architecture (2 conv + pooling blocks, 3 dense layers)	MNIST	99.0% accuracy, loss 0.049
AlexNet.ipynb	10-class image classification with an AlexNet-style CNN (5 conv blocks with batch norm)	CIFAR-10 (Kaggle)	72.5% accuracy, loss 1.058

All three build up in complexity from a simple custom CNN → the classic LeNet-5 → a deeper AlexNet-style network, illustrating how architecture depth and design choices (batch norm, more filters) affect classification performance on different datasets.

🛠️ Tech Stack
Language: Python
Core libraries: NumPy, Matplotlib
Computer vision: OpenCV (cv2)
Deep learning: TensorFlow / Keras
Object detection: Ultralytics YOLOv8
Data: opendatasets (Kaggle dataset downloads), scikit-learn (confusion_matrix)
🚀 Getting Started
bash
git clone https://github.com/devanshu6738/Computer_Vision.git
cd Computer_Vision
pip install numpy matplotlib opencv-python tensorflow keras ultralytics opendatasets scikit-learn

Then open any notebook in Jupyter or Google Colab (the Kaggle-based notebooks — basic_cnn.ipynb, AlexNet.ipynb — download their datasets automatically via opendatasets, which requires a Kaggle API key).

Notebooks that use cv2.VideoCapture(0) (VideoProcessing.ipynb, FeatureDetection.ipynb, faceDetection.ipynb, objectDetection.ipynb) need a local webcam and won't work as-is in a headless/Colab environment.

🔧 Possible Improvements
Add a requirements.txt pinning library versions.
Move sample images and pretrained weights into a dedicated assets/ or weights/ folder.
Add markdown explanations/theory notes above each code cell (several notebooks currently jump straight into code).
Save training-history plots (accuracy/loss curves) for the CNN notebooks instead of only printing final metrics.
Extract the manual convolution/Sobel implementations into reusable functions/module for comparison against OpenCV's built-in equivalents.
🙋 Author

Devanshu — @devanshu6738
