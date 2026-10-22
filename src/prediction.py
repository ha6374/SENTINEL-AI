from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.audio_preprocessing import extract_features


MODEL_PATH = Path("models/best_audio_cnn.pth")
DEVICE = torch.device("cpu")
FAKE_THRESHOLD = 0.90


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 8)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = AudioCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_audio(audio_path, model=None):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if model is None:
        model = load_model()

    features = extract_features(audio_path)

    if features.shape != (64, 313):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    input_array = np.ascontiguousarray(
        features[None, None, :, :],
        dtype=np.float32,
    )

    input_tensor = torch.from_numpy(input_array).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    real_probability = float(probabilities[0].item())
    fake_probability = float(probabilities[1].item())

    if fake_probability >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = fake_probability
    else:
        prediction = "Real"
        confidence = real_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "threshold": FAKE_THRESHOLD,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect potentially spoofed audio."
    )
    parser.add_argument(
        "audio_path",
        help="Path to the audio file to analyze",
    )
    args = parser.parse_args()

    result = predict_audio(args.audio_path)

    print("\nAUDIO ANALYSIS RESULT")
    print("-" * 35)
    print("Prediction:", result["prediction"])
    print(f"Model confidence: {result['confidence']:.2%}")
    print(f"Real probability: {result['real_probability']:.2%}")
    print(f"Fake probability: {result['fake_probability']:.2%}")
    print(f"Fake threshold: {result['threshold']:.2f}")
    print("\nNote: This is a model estimate, not a guarantee.")
