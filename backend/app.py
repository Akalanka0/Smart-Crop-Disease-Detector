"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🌿  CROP DISEASE DETECTOR  –  Flask API Backend                   ║
║                                                                              ║
║  Serves:                                                                     ║
║    GET  /            → frontend/index.html                                  ║
║    POST /predict     → run inference on uploaded image                      ║
║    GET  /classes     → list all trained class names                         ║
║    GET  /health      → server status                                         ║
║                                                                              ║
║  Run from the project root:                                                  ║
║    python backend/app.py                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import io
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Resolve paths relative to the project root ───────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
from torchvision import transforms, models

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ←  adjust thresholds if needed
# ══════════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.85   # Raised to extremely strict levels (85%)
ENTROPY_THRESHOLD    = 1.50   # Lower = Stricter (Reject if the model is 'confused')
MODEL_PATH           = os.path.join(ROOT, "model_best.pth")
CLASS_NAMES_PATH     = os.path.join(ROOT, "class_names.json")

# CLIP Guard Settings (Zero-shot semantic verification)
USE_CLIP_GUARD       = True
CLIP_PLANT_THRESHOLD = 0.40  # Lowered for better recall (40%)

NON_PLANT_PROMPTS = [
    "a photo of a car",
    "a photo of a person",
    "a photo of a building",
    "a photo of furniture or household objects",
    "a photo of a chair or table",
    "a photo of a person's hand",
    "an indoor room scene",
    "a digital screenshot or text",
    "random color texture or noise",
    "a blurry or irrelevant image",
]
# ══════════════════════════════════════════════════════════════════════════════

# ── Load class names ──────────────────────────────────────────────────────────
CLASSES = []
PLANT_PROMPTS = []

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASSES = json.load(f)
    
    # Dynamic CLIP Prompts: Focuses CLIP exactly on the crops the model knows about
    PLANT_PROMPTS = [
        f"a photo of a {c.replace('_', ' ')} leaf" 
        for c in CLASSES if c.lower() != "background"
    ] + [
        "a close-up of agricultural foliage",
        "a photo of a diseased crop leaf",
        "a photo of a healthy plant leaf"
    ]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Disease Model ────────────────────────────────────────────────────────
def load_disease_model():
    if not os.path.exists(MODEL_PATH): return None
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


# ── Load CLIP Model (Optional/Dynamic) ────────────────────────────────────────
def load_clip_model():
    if not USE_CLIP_GUARD: return None, None
    try:
        import clip
        print("📥  Loading CLIP model (ViT-B/32)...")
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        return model, preprocess
    except ImportError:
        print("⚠️  CLIP not installed. 'Plant Guard' disabled.")
        return None, None


DISEASE_MODEL = None
CLIP_MODEL, CLIP_PREPROCESS = load_clip_model()

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
    try:
        DISEASE_MODEL = load_disease_model()
        print(f"✅  Backend ready  |  device: {DEVICE}  |  classes: {len(CLASSES)}")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
else:
    print("📢  [TEMPLATE MODE] Model or class names not found. Server is running, but /predict will require training.")

# ── Image transform for Disease Classifier ───────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Plant Guard Check ─────────────────────────────────────────────────────────
def is_plant_image(pil_image) -> tuple[bool, float]:
    if not CLIP_MODEL or not CLIP_PREPROCESS:
        return True, 1.0  # Pass if CLIP is disabled
    
    import clip
    all_prompts  = PLANT_PROMPTS + NON_PLANT_PROMPTS
    text_tokens  = clip.tokenize(all_prompts).to(DEVICE)
    image_tensor = CLIP_PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat  = CLIP_MODEL.encode_image(image_tensor)
        text_feat = CLIP_MODEL.encode_text(text_tokens)
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        probs = (img_feat @ text_feat.T).squeeze(0).softmax(dim=0).cpu()

    plant_score = probs[: len(PLANT_PROMPTS)].sum().item()
    return plant_score >= CLIP_PLANT_THRESHOLD, plant_score


def run_prediction(pil_image: Image.Image) -> dict:
    if DISEASE_MODEL is None:
        return {
            "status": "error",
            "error": "Model not found on server. Please run 'python train.py' on the server machine to generate the model and class metadata."
        }
    
    # 1. Plant Guard (CLIP)
    is_plant, plant_score = is_plant_image(pil_image)
    if not is_plant:
        return {
            "status":      "not_a_plant",
            "plant_score": round(plant_score * 100, 2),
            "error":       "This doesn't look like a leaf or plant. Please upload a clear photo of agricultural foliage."
        }

    # 2. Disease Classifier
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs       = DISEASE_MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_idx = probabilities.max(0)

    entropy = -(probabilities * (probabilities + 1e-9).log()).sum().item()
    top5 = sorted(
        {c: round(p.item() * 100, 2) for c, p in zip(CLASSES, probabilities)}.items(),
        key=lambda x: x[1], reverse=True
    )[:5]

    predicted_class = CLASSES[predicted_idx.item()]

    # 3. Handle explicit "Background" class
    if predicted_class.lower() == "background":
        return {
            "status":      "not_a_plant",
            "plant_score": round(plant_score * 100, 2),
            "error":       "This image was classified as 'Background / Not a Leaf'. Please upload a clearer photo of a crop leaf."
        }

    return {
        "status":      "uncertain" if (confidence.item() < CONFIDENCE_THRESHOLD
                                       or entropy > ENTROPY_THRESHOLD) else "ok",
        "class":       predicted_class,
        "label":       predicted_class.replace("_", " "),
        "confidence":  round(confidence.item() * 100, 2),
        "entropy":     round(entropy, 4),
        "top5":        top5,
        "plant_score": round(plant_score * 100, 2),
    }


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(ROOT, "frontend"),
    static_url_path="/",
)
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {str(e)}"}), 400

    try:
        return jsonify(run_prediction(pil_image))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": CLASSES, "count": len(CLASSES)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "device":       str(DEVICE),
        "classes":      len(CLASSES),
        "clip_guard":   bool(CLIP_MODEL),
        "model_path":   MODEL_PATH,
    })


if __name__ == "__main__":
    print("\n🌿  Crop Disease Detector API")
    print(f"   Open browser at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
