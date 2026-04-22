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
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="AgriVision Plant Disease Detection")

# Load model
MODEL_PATH = "."
MODEL_ID = "mesabo/agri-plant-disease-resnet50"

print("Loading model...")
try:
    # Try loading locally first
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
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    with open(index_path, "r") as f:
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
        
        # Safe lookup for both int and str keys
        label = id2label.get(idx) or id2label.get(str(idx))
        if not label:
            # Final fallback: numeric search
            for k, v in id2label.items():
                if int(k) == idx:
                    label = v
                    break
            if not label: label = "Unknown"
        
        confidence = top_prob.item()
        
        # Parse label (format is usually "Plant___Disease")
        if "___" in label:
            plant, disease = label.split("___")
        else:
            plant = "Crop"
            disease = label
            
        plant_name = plant.replace("_", " ")
        disease_name = disease.replace("_", " ")
        status = "Healthy" if "healthy" in disease.lower() else "Diseased"
        
        explanation = ""
        if status == "Diseased" and groq_client:
            try:
                prompt = f"Explain the plant disease '{disease_name}' affecting '{plant_name}'. Provide a brief description, causes, and treatment/prevention steps in a concise, helpful way for a farmer."
                completion = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are an expert plant pathologist and agricultural advisor. Provide concise, practical advice."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                )
                explanation = completion.choices[0].message.content
            except Exception as ge:
                print(f"Groq API error: {ge}")
                explanation = "Could not fetch detailed explanation at this time."
        elif status == "Healthy":
            explanation = f"Your {plant_name} appears to be healthy! Keep up the good work with regular watering and monitoring."
        else:
            explanation = "Please provide a Groq API key in the .env file to see detailed disease explanations."

        return {
            "plant": plant_name,
            "disease": disease_name,
            "confidence": f"{confidence:.2%}",
            "status": status,
            "explanation": explanation
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

if __name__ == "__main__":
    # Get port from environment variable (default to 8000 for local dev)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
