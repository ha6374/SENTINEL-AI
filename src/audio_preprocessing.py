
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION


def load_audio(file_path):
    """
    Load an audio file, convert it to mono,
    and resample it to 16 kHz.
    """
    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Make every audio sample exactly 5 seconds long.
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    return audio.astype(np.float32)


def extract_features(file_path):
    """
    Extract a log-Mel spectrogram from an audio file.
    Returns a NumPy array for model training.
    """
    audio = load_audio(file_path)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )

    # Normalize feature values.
    log_mel = np.clip(log_mel, -80.0, 0.0) / 80.0

    return log_mel.astype(np.float32)


def get_audio_info(file_path):
    """
    Return basic information about an audio file.
    """
    info = sf.info(str(file_path))

    return {
        "filename": Path(file_path).name,
        "sample_rate": info.samplerate,
        "duration_seconds": round(info.duration, 2),
        "channels": info.channels,
        "format": info.format
    }

