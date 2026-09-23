from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.model import predict_image


app = FastAPI(
    title="AI Image Detector API",
    description="Detect whether an image is REAL or AI-GENERATED",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# -----------------------------
# Image prediction
# -----------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # Make sure uploaded file is an image
    if not image.content_type or not image.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:

        # Read uploaded file
        contents = await image.read()

        # Convert to PIL image
        img = Image.open(
            io.BytesIO(contents)
        )

        # Make prediction
        prediction, confidence = predict_image(img)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )
