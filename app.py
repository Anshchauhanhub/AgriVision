import os
import io
import torch
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import ResNetForImageClassification
import uvicorn

app = FastAPI(title="AgriVision Plant Disease Detection")

# Load model
MODEL_PATH = "."
MODEL_ID = "mesabo/agri-plant-disease-resnet50"

print("Loading model...")
try:
    model = ResNetForImageClassification.from_pretrained(MODEL_PATH)
    print("Loaded model from local files.")
except Exception as e:
    print(f"Local model load failed: {e}. Falling back to Hugging Face Hub...")
    try:
        model = ResNetForImageClassification.from_pretrained(MODEL_ID)
        print("Loaded model from Hugging Face Hub.")
    except Exception as e2:
        print(f"Failed to load model from Hub: {e2}")
        raise RuntimeError("Could not load model")

model.eval()

# Manual preprocessing for ResNet50 (Standard)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if not os.path.exists("index.html"):
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_batch)
            logits = outputs.logits
            
        # Post-process results
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.max(probabilities, dim=-1)
        
        # Retrieve label safely
        idx = top_idx.item()
        id2label = model.config.id2label
        if idx in id2label:
            label = id2label[idx]
        elif str(idx) in id2label:
            label = id2label[str(idx)]
        else:
            # Try converting all keys to int for comparison
            label = "Unknown"
            for k, v in id2label.items():
                if int(k) == idx:
                    label = v
                    break
        
        confidence = top_prob.item()
        
        # Parse label (format is usually "Plant___Disease")
        if "___" in label:
            plant, disease = label.split("___")
        else:
            plant = "Crop"
            disease = label
            
        return {
            "plant": plant.replace("_", " "),
            "disease": disease.replace("_", " "),
            "confidence": f"{confidence:.2%}",
            "status": "Healthy" if "healthy" in disease.lower() else "Diseased"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
