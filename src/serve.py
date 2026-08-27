import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import CIFAR10_CLASSES, get_model

# module-level state for the loaded model
_model = None
_device = None

# must match the normalization used during training
_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def _load_model():
    global _model, _device

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    model_name = os.environ.get("MODEL_NAME", "classifier_v1.pt")
    checkpoint_path = Path(checkpoint_dir) / model_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read architecture from config if available
    config_path = Path(os.environ.get("CONFIG_PATH", "/app/configs/training_config.yaml"))
    architecture = "resnet18"
    num_classes = 10
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        architecture = cfg.get("model", {}).get("architecture", "resnet18")
        num_classes = cfg.get("model", {}).get("num_classes", 10)

    _model = get_model(architecture=architecture, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", "?")
    print(f"Model loaded: {checkpoint_path} (epoch={epoch}, val_acc={val_acc})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except FileNotFoundError as e:
        print(f"WARNING: {e}. /predict will be unavailable until a checkpoint is present.")
    yield


app = FastAPI(title="CIFAR-10 Classifier API", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    class_probs = {cls: round(p, 4) for cls, p in zip(CIFAR10_CLASSES, probs)}
    top_class = max(class_probs, key=class_probs.get)

    return JSONResponse({
        "predicted_class": top_class,
        "confidence": class_probs[top_class],
        "class_probabilities": class_probs,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
