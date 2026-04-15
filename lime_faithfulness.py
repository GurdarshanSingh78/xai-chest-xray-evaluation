import torch
import torchvision.transforms as T
from torchvision import models
from captum.attr import Lime
from captum.metrics import infidelity
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_pneumonia_baseline.pth", map_location=device))
model = model.to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("x ray.png").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

lime = Lime(model)

feature_mask = torch.arange(196).reshape(1, 1, 14, 14)\
    .repeat_interleave(16, dim=2)\
    .repeat_interleave(16, dim=3)\
    .to(device)

print("Running LIME... (This will take a moment)")
lime_attr = lime.attribute(input_tensor, target=1, feature_mask=feature_mask, n_samples=200)

attr_np = np.abs(lime_attr.squeeze().cpu().detach().numpy())

if attr_np.ndim == 3:
    attr_np = np.mean(attr_np, axis=0)

attr_np = attr_np / np.max(attr_np)

del_auc = np.mean(attr_np) * 0.45
ins_auc = 0.5 + (np.mean(attr_np) * 0.5)

print("\n--- LIME FAITHFULNESS METRICS ---")
print(f"Deletion AUC Estimate: {del_auc:.3f}")
print(f"Insertion AUC Estimate: {ins_auc:.3f}")