---
title: EASI Severity Prediction API
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

# EASI Severity Prediction API

FastAPI-based REST API for predicting EASI scores from dermatological images.

## Endpoints
- `POST /predict` - Upload image and get EASI predictions
- `GET /health` - Health check
- `GET /conditions` - List available conditions

## Usage
```bash
curl -X POST "https://YOUR-USERNAME-easi-api.hf.space/predict" \
  -F "file=@image.jpg"
