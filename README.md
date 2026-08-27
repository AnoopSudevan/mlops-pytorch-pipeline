# mlops-pytorch-pipeline

End-to-end MLOps pipeline for training and serving a CIFAR-10 image classifier using PyTorch, Docker, and Kubernetes.

## Architecture

```
  GitHub Actions CI
        │
        ▼
  Docker Build
  ┌────────────────────────┐
  │  mlops-train:v1        │
  │  mlops-serve:v1        │
  └──────────┬─────────────┘
             │
             ▼
  Kubernetes Cluster (ml-training namespace)
  ┌─────────────────────────────────────┐
  │                                     │
  │  ┌─────────────────────────────┐    │
  │  │  Job: pytorch-training      │    │
  │  │  - ConfigMap (config yaml)  │    │
  │  │  - PVC: training-data-pvc   │    │
  │  │  - PVC: model-checkpoints   │    │
  │  └────────────┬────────────────┘    │
  │               │ writes checkpoint   │
  │  ┌────────────▼────────────────┐    │
  │  │  Deployment: model-serving  │    │
  │  │  - 2 replicas (HPA: 2–6)   │    │
  │  │  - PVC: checkpoints (ro)   │    │
  │  │  - POST /predict            │    │
  │  │  - GET  /health             │    │
  │  └────────────┬────────────────┘    │
  │  ┌────────────▼────────────────┐    │
  │  │  Service: model-serving     │    │
  │  │  ClusterIP port 80 → 8080  │    │
  │  └─────────────────────────────┘    │
  └─────────────────────────────────────┘
```

## Project Structure

```
mlops-pytorch-pipeline/
├── src/
│   ├── model.py        # ResNet-18 adapted for CIFAR-10, plus SimpleCNN fallback
│   ├── dataset.py      # Data loading, augmentation, CIFAR-10 / Fashion-MNIST
│   ├── train.py        # Training loop, early stopping, JSON-lines logging
│   └── serve.py        # FastAPI inference server (/predict, /health)
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train   # Multi-stage training image
│   └── Dockerfile.serve   # Non-root serving image with HEALTHCHECK
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Local Setup

```bash
# Clone and set up environment
git clone https://github.com/AnoopSudevan/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements/train.txt

# Run training locally (downloads CIFAR-10 automatically)
python src/train.py --config configs/training_config.yaml

# Run tests
pip install pytest
pytest tests/ -v
```

## Docker

```bash
# Build images
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Train (mounts local data/ and checkpoints/ directories)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serve the trained model
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test the prediction endpoint
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
curl http://localhost:8080/health
```

## Kubernetes

```bash
# 1. Create namespace, storage, and config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml

# 2. Run training job
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/pytorch-training -n ml-training --timeout=3600s
kubectl logs job/pytorch-training -n ml-training

# 3. Deploy the serving layer
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Verify everything is healthy
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 5. Test prediction endpoint via port-forward
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Model Details

**Architecture:** ResNet-18 modified for 32×32 CIFAR-10 images
- First `conv1` changed to `kernel_size=3, stride=1, padding=1` (no spatial downsampling)
- `MaxPool` replaced with `Identity` layer
- Final FC head outputs 10 class logits

**Training:** Adam optimizer + cosine annealing LR schedule + early stopping

**Expected performance:** ~92–93% validation accuracy on CIFAR-10 after ~15 epochs

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status": "ok"}` if model is loaded, 503 otherwise |
| `/predict` | POST | Accepts `image` form field, returns class probabilities |

**Predict response example:**
```json
{
  "predicted_class": "automobile",
  "confidence": 0.9832,
  "class_probabilities": {
    "airplane": 0.0021, "automobile": 0.9832, ...
  }
}
```
