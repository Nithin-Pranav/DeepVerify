from django.shortcuts import render
from django.conf import settings
import os
import uuid
from datetime import datetime
from .utils import predict_image, predict_video, predict_audio, final_deepverify_prediction


def home(request):
    history = request.session.get('history', [])
    return render(request, 'home.html', {'history': history})

def predict(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        video_file = request.FILES.get('video')
        audio_file = request.FILES.get('audio')

        if not any([image_file, video_file, audio_file]):
            return render(request, 'home.html', {'error': 'Please upload at least one file.'})

        # 🔹 Option 1: Prevent very large uploads (optional but recommended)
        MAX_SIZE_MB = 20  # 20 MB limit per file
        if image_file and image_file.size > MAX_SIZE_MB * 1024 * 1024:
            return render(request, 'home.html', {'error': f'Image too large (max {MAX_SIZE_MB}MB)'})
        if video_file and video_file.size > MAX_SIZE_MB * 1024 * 1024:
            return render(request, 'home.html', {'error': f'Video too large (max {MAX_SIZE_MB}MB)'})
        if audio_file and audio_file.size > MAX_SIZE_MB * 1024 * 1024:
            return render(request, 'home.html', {'error': f'Audio too large (max {MAX_SIZE_MB}MB)'})

        # Save with unique names
        image_score = None
        video_score = None
        audio_score = None

        if image_file:
            filename = f"{uuid.uuid4()}_{image_file.name}"
            path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'images', filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
            image_score = predict_image(path)

        if video_file:
            filename = f"{uuid.uuid4()}_{video_file.name}"
            path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'videos', filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb+') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            video_score = predict_video(path)

        if audio_file:
            filename = f"{uuid.uuid4()}_{audio_file.name}"
            path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'audio', filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb+') as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)
            audio_score = predict_audio(path)

        # Final fusion
        result = final_deepverify_prediction(
            image_score=image_score,
            video_visual_score=video_score,
            audio_score=audio_score
        )

        # ✅ Define probabilities for chart
        if result["label"] == "Fake":
            fake_prob = result["confidence"]
            real_prob = 1 - result["confidence"]
        elif result["label"] == "Real":
            real_prob = result["confidence"]
            fake_prob = 1 - result["confidence"]
        else:
            fake_prob = 0.5
            real_prob = 0.5

        # ✅ Session-based history (safe)
        history = request.session.get('history', [])
        history.append({
            'label': result['label'],
            'confidence': result['confidence'],
            'details': result['details'],
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        request.session['history'] = history[-10:]

        context = {
            'result': result,
            'history': history,
            'fake_prob': round(fake_prob, 3),
            'real_prob': round(real_prob, 3),
            'image_score': round(image_score, 3) if image_score is not None else None,
            'video_score': round(video_score, 3) if video_score is not None else None,
            'audio_score': round(audio_score, 3) if audio_score is not None else None,
        }

        return render(request, 'result.html', context)

    return render(request, 'home.html', {
        'history': request.session.get('history', [])
    })