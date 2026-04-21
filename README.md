# AgriVision 🌿

**AgriVision** is a premium, AI-powered web application designed for real-time plant disease detection. Developed as a PBL (Project Based Learning) project, it leverages deep learning to help farmers and researchers identify crop health issues instantly through image analysis.

![AgriVision Preview](https://raw.githubusercontent.com/Anshchauhanhub/AgriVision/pbl-project/static/preview.png) *(Note: Add a screenshot of your app here!)*

## 🚀 Features

- **Premium UI/UX**: A modern, dark-themed interface featuring glassmorphism effects and smooth micro-animations.
- **Deep Learning Core**: Powered by a **ResNet50** model (`mesabo/agri-plant-disease-resnet50`) from Hugging Face.
- **Instant Diagnosis**: Drag-and-drop leaf images for real-time identification of 38 different plant health categories.
- **Responsive Design**: Optimized for both desktop and mobile viewing.

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **ML Frameworks**: PyTorch, Transformers, Safetensors
- **Image Processing**: Torchvision, Pillow

## 🌾 Supported Crops

The AI can detect diseases across 14 different plant species:
- Apple
- Blueberry
- Cherry
- Corn (Maize)
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Anshchauhanhub/AgriVision.git
   cd AgriVision
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:8000`.

## 🧪 Model Details

- **Model Architecture**: ResNet50
- **Dataset**: Trained on the PlantVillage dataset.
- **Format**: Safetensors

## 👨‍💻 Author

**Ansh Chauhan**  
GitHub: [@Anshchauhanhub](https://github.com/Anshchauhanhub)

---
*Created for PBL Project 2026*
