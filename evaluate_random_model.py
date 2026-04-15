import torch
import torchvision.transforms as T
from torchvision import models
from captum.attr import IntegratedGradients, LayerGradCam
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing model with completely randomized weights...")
random_model = models.resnet50(weights=None)
random_model.fc = torch.nn.Linear(random_model.fc.in_features, 2)
random_model = random_model.to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("x ray.png").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

ig = IntegratedGradients(random_model)
gradcam = LayerGradCam(random_model, random_model.layer4)

print("Extracting Saliency Maps...")
ig_attr, _ = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
gc_attr = gradcam.attribute(input_tensor, target=1)

def process_attr(attr):
    arr = attr.squeeze().cpu().detach().numpy()
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    arr = np.maximum(arr, 0)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    return arr

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img.resize((224,224)))
axes[0].set_title("Original X-Ray")

axes[1].imshow(process_attr(ig_attr), cmap='magma')
axes[1].set_title("IG (Randomized Model)")

axes[2].imshow(process_attr(gc_attr), cmap='jet')
axes[2].set_title("Grad-CAM (Randomized Model)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("random_model_check.png", bbox_inches='tight', dpi=300)

print("✅ SUCCESS: Saved extreme sanity check as 'random_model_check.png'")