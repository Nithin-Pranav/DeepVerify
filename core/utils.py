import torch
import timm
import numpy as np
import librosa
import soundfile as sf
from django.conf import settings
import os
import cv2
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip  # ← Fixed for moviepy v2+
import tempfile

# Disable gradients globally
torch.set_grad_enabled(False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
MODELS_DIR = os.path.join(settings.BASE_DIR, "models")
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "best_image_model.pth")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "best_audio_crnn.pth")

# 🔧 Consistent class index (used for both models)
FAKE_CLASS_INDEX = 0  # 0 = Fake, 1 = Real (confirmed from your training)

# -----------------------------
# 1. IMAGE MODEL
# -----------------------------
try:
    image_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    image_model.to(device)
    image_model.eval()
    print("✅ Image model loaded successfully")
except Exception as e:
    print("❌ Failed to load image model:", e)
    image_model = None

# Image transform
from torchvision import transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    if image_model is None:
        return 0.5
    
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        tensor = image_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = image_model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            fake_prob = prob[FAKE_CLASS_INDEX].item()  # ← Fixed: consistent
        
        return fake_prob
    except Exception as e:
        print("Image prediction error:", e)
        return 0.5

# -----------------------------
# 2. AUDIO MODEL
# -----------------------------
class CRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,4)),
        )
        self.rnn = torch.nn.GRU(1024, 48, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(96, 2)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, T = x.shape
        x = x.reshape(B, C * H, T)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

try:
    audio_model = CRNN()
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
    audio_model.to(device)
    audio_model.eval()
    print("✅ Audio model loaded successfully")
except Exception as e:
    print("❌ Failed to load audio model:", e)
    audio_model = None

# Audio config
SR = 16000
N_MELS = 64
HOP_LENGTH = 160
N_FFT = 512
FMAX = 8000
MAX_TIME_STEPS = 400

def predict_audio(audio_path):
    if audio_model is None:
        return 0.5
    
    try:
        y, sr = sf.read(audio_path)
        if len(y) < SR * 0.5:
            return 0.5
        
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if log_mel_spec.shape[1] < MAX_TIME_STEPS:
            pad_width = MAX_TIME_STEPS - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :MAX_TIME_STEPS]
        
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
        log_mel_spec = log_mel_spec[np.newaxis, :, :]
        
        tensor = torch.tensor(log_mel_spec).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = audio_model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            fake_prob = prob[FAKE_CLASS_INDEX].item()  # ← Fixed: consistent with image
        
        return fake_prob
    except Exception as e:
        print("Audio prediction error:", e)
        return 0.5

# -----------------------------
# 3. VIDEO PLACEHOLDER
# -----------------------------

def predict_video(video_path):
    """
    Full video prediction:
    - Extract frames → image model (visual score)
    - Extract audio → audio model (audio score)
    - Average both for final fake probability
    """
    visual_score = 0.5
    audio_score = 0.5

    try:
        # -----------------------------
        # 1. Visual: Frame extraction (FIX 3 — safety check)
        # -----------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file:", video_path)
            return 0.5

        visual_scores = []
        frame_count = 0
        max_frames = 20

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                tensor = image_transform(pil_img).unsqueeze(0).to(device)
                # torch.no_grad() redundant but safe — keep it
                with torch.no_grad():
                    output = image_model(tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    visual_scores.append(prob[FAKE_CLASS_INDEX].item())
            frame_count += 1
        cap.release()

        if visual_scores:
            visual_score = np.mean(visual_scores)
        print(f"Visual score: {visual_score:.3f} (from {len(visual_scores)} frames)")

        # -----------------------------
        # 2. Audio: Extract and predict (FIX 2 — safe close)
        # -----------------------------
        video_clip = None
        tmp_audio_path = None
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            if audio_clip is None:
                print("No audio track in video")
                audio_score = 0.5
            else:
                # Create temp file
                tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_audio_path = tmp_file.name
                tmp_file.close()

                audio_clip.write_audiofile(tmp_audio_path, verbose=False, logger=None)
                audio_score = predict_audio(tmp_audio_path)
                print(f"Audio score: {audio_score:.3f}")
        except Exception as e:
            print("Audio extraction error:", e)
            audio_score = 0.5
        finally:
            # Always close video clip
            if video_clip:
                video_clip.close()
            # Clean up temp audio file
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                try:
                    os.unlink(tmp_audio_path)
                except:
                    pass

        # -----------------------------
        # 3. Final score
        # -----------------------------
        final_score = (visual_score + audio_score) / 2
        print(f"Final video fake probability: {final_score:.3f}")

        return final_score

    except Exception as e:
        print("Video prediction critical error:", e)
        return 0.5

# -----------------------------
# 4. MULTIMODAL FUSION
# -----------------------------
def final_deepverify_prediction(
    image_score=None,
    video_visual_score=None,
    audio_score=None,
    min_confidence=0.75,
):
    scores = []
    if image_score is not None:
        scores.append(("Image", image_score))
    if video_visual_score is not None:
        scores.append(("Video", video_visual_score))
    if audio_score is not None:
        scores.append(("Audio", audio_score))
    
    if not scores:
        return {"label": "Uncertain", "confidence": 0.0, "details": "No prediction"}
    
    avg_fake = np.mean([s[1] for s in scores])
    num = len(scores)
    fake_votes = sum(1 for _, s in scores if s > 0.5)
    real_votes = num - fake_votes
    
    if fake_votes > real_votes and avg_fake >= min_confidence:
        label = "Fake"
        confidence = avg_fake
    elif real_votes > fake_votes and (1 - avg_fake) >= min_confidence:
        label = "Real"
        confidence = 1 - avg_fake
    else:
        label = "Uncertain"
        confidence = max(avg_fake, 1 - avg_fake)
    
    confidence = min(max(confidence, 0.0), 1.0)
    
    breakdown = [f"{mod}: {'Fake' if s > 0.5 else 'Real'} ({s if s > 0.5 else 1-s:.3f})" for mod, s in scores]
    
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "details": f"Modalities: {num} | {', '.join(breakdown)}"
    }