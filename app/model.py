import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Use GPU if available, otherwise CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Create EfficientNet-B0
# -----------------------------
model = models.efficientnet_b0(weights=None)

# Binary classification:
# 0 = REAL
# 1 = AI_GENERATED
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/efficientnet_new_class_balanced_best.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image: Image.Image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    real_probability = probabilities[0][0].item()
    ai_probability = probabilities[0][1].item()

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    if prediction == 1:

        label = "AI_GENERATED"
        confidence = ai_probability

    else:

        label = "REAL"
        confidence = real_probability

    return label, confidence
