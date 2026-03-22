import torch
from torchvision import transforms
from PIL import Image

from models.cnn_model import CNNModel

# 1. Classes (IMPORTANT — must match training)
classes = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# 2. Load model
model = CNNModel(num_classes=len(classes))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 3. Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 4. Load test image (CHANGE THIS PATH)
image_path = "dataset/Tomato_Early_blight/0a0c1c.jpg"  # example
image = Image.open(image_path).convert("RGB")

# 5. Preprocess
image = transform(image).unsqueeze(0)

# 6. Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# 7. Show result
print("Predicted class:", classes[predicted.item()])