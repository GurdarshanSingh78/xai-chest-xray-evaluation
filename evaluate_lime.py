import torch
import torchvision.transforms as T
from torchvision import models
from captum.attr import Lime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

print("Initializing LIME explainer...")
lime = Lime(model)

feature_mask = torch.arange(196).reshape(1, 1, 14, 14).repeat_interleave(16, dim=2).repeat_interleave(16, dim=3).to(device)

print("Running LIME perturbations... (This takes a few seconds)")
lime_attr = lime.attribute(input_tensor, target=1, feature_mask=feature_mask, n_samples=200)

lime_np = lime_attr.squeeze().cpu().detach().numpy()
lime_np = np.mean(lime_np, axis=0)
lime_np = np.maximum(lime_np, 0)

if np.max(lime_np) > 0:
    lime_np = lime_np / np.max(lime_np)

plt.figure(figsize=(6, 6))
plt.imshow(img.resize((224, 224)))
plt.imshow(lime_np, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title("LIME Saliency Map")
plt.savefig("lime_output.png", bbox_inches='tight', pad_inches=0)

print("Saved LIME heatmap as 'lime_output.png'")