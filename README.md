Emotion Recognition System Using Facial Expressions 
A real-time Deep Learning application that detects human emotions through a laptop webcam. This project uses a custom-trained Convolutional Neural Network (CNN) built with TensorFlow/Keras and OpenCV for computer vision processing.

Features
Real-time Detection: Process video frames at 30+ FPS.

Custom CNN Architecture: Optimized for CPU-based inference on laptops.

Pre-processing Pipeline: Includes Histogram Equalization to handle varied lighting conditions and face cropping for high accuracy.

Data Augmentation: Trained with image rotation, zooming, and flipping to prevent overfitting.

Tech Stack
Language: Python 3.9+

Deep Learning: TensorFlow, Keras

Computer Vision: OpenCV (Haar Cascades)

Data Handling: NumPy, Pandas, SciPy

Dataset
The model is trained on the FER-2013 dataset (Facial Expression Recognition 2013), which contains ~30,000 grayscale images categorized into 7 emotions:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

📂 Project Structure
Plaintext
.
├── archive/             # Dataset (Train/Test folders)
├── vision_engine.py     # Main real-time inference script
├── train_model.py       # CNN training script
├── emotion_model.h5     # Trained model weights
└── README.md            # Project documentation
⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/yourusername/facial-expression-recognition.git
cd facial-expression-recognition
Set up Virtual Environment:

Bash
python -m venv .venv
.\.venv\Scripts\activate
Install Dependencies:

Bash
pip install opencv-python tensorflow numpy scipy
Run the Application:

Bash
python vision_engine.py
Methodology
To solve the common issue of "rubbish answers" in facial recognition, this project implements:

Region of Interest (ROI): Isolating the face to ignore background noise.

Histogram Equalization: Normalizing brightness to improve feature detection in low-light environments.

Batch Normalization: Used in the CNN layers to stabilize the learning process.
