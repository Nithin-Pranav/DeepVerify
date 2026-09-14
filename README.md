# DeepVerify - Multimodal Deepfake Detection System

DeepVerify is a multimodal deepfake detection system developed to analyze images, audio, and video content and identify whether the provided media is real or manipulated.

The system uses deep learning models to analyze different media modalities individually. Image analysis is performed using an EfficientNet-B0-based model, while audio analysis uses a CNN-based model. Video analysis is performed by extracting and analyzing video frames using the image deepfake detection model.

When multiple media modalities are analyzed, the system can combine their prediction probabilities to provide an overall result.

The final prediction is presented as:

- Real
- Fake
- Uncertain

---

## Features

- Image deepfake detection
- Audio deepfake detection
- Video deepfake detection using extracted frames
- Multimodal media analysis
- Prediction probability fusion across supported modalities
- Deep learning-based detection models
- Django-based web application
- User-friendly interface for uploading and analyzing media files

---

## Technologies Used

### Backend

- Python
- Django

### Deep Learning and Machine Learning

- PyTorch
- EfficientNet-B0
- Convolutional Neural Network (CNN)
- NumPy

### Media Processing

- OpenCV
- Librosa
- FFmpeg

### Frontend

- HTML
- CSS
- Django Templates

---

## Project Structure

```text
DeepVerify/
│
├── core/                       # Main Django application
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── utils.py
│   └── views.py
│
├── deepverify/                 # Django project configuration
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── models/                     # Trained deep learning models
│   ├── best_audio_cnn.pth
│   └── best_image_model.pth
│
├── templates/                  # HTML templates
│   ├── home.html
│   └── result.html
│
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md

```
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nithin-Pranav/DeepVerify.git
```

### 2. Navigate to the Project Directory

```bash
cd DeepVerify
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows

```bash
venv\Scripts\activate
```
macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Apply Django Migrations

```bash
python manage.py migrate
```

Start the Development Server

```bash
python manage.py runserver
```

Open the application in your browser:

http://127.0.0.1:8000/

## How It Works

### Image Analysis

Uploaded images are processed using an EfficientNet-B0-based deep learning model. The model analyzes the image and generates a prediction indicating whether the content is likely to be real or manipulated.

### Audio Analysis

Audio files are processed to extract relevant audio features. These features are then analyzed using a CNN-based deep learning model to detect potential audio manipulation.

### Video Analysis

DeepVerify does not use a separate video-specific deep learning model.

Instead, the system processes videos by extracting relevant frames and analyzing those frames using the trained image deepfake detection model.

The frame-level predictions are then used to determine the overall prediction for the uploaded video.

### Multimodal Analysis

When multiple supported media modalities are analyzed, DeepVerify can combine their prediction probabilities to generate an overall result.

The system processes the available image, audio, and video inputs separately and uses their analysis results for multimodal prediction.

The final output can indicate:

- **Real** – The analyzed media is predicted to be authentic.
- **Fake** – The analyzed media is predicted to be manipulated.
- **Uncertain** – The prediction does not meet the required confidence threshold for a clear Real or Fake classification.

## Supported Media Types

The system supports analysis of:

- Images
- Audio files
- Video files

## Model Files

The trained deep learning models are stored in the `models/` directory.

### Audio Model

`best_audio_cnn.pth`

This model is used for audio deepfake detection.

### Image Model

`best_image_model.pth`

This model is used for:

- Image deepfake detection
- Video deepfake detection through frame analysis

The same image model is integrated into the video analysis pipeline, so a separate video model is not required.

## Requirements

All required Python dependencies are listed in:

`requirements.txt`

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Future Improvements

Possible future improvements include:

- Improve deepfake detection accuracy
- Train additional and more robust deep learning models
- Improve video-level analysis
- Support additional media formats
- Enhance multimodal prediction fusion
- Deploy the application to a cloud platform
- Add user authentication and account management
- Improve the user interface and user experience
