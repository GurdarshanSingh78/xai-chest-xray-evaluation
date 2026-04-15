import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from captum.attr import IntegratedGradients, LayerGradCam, Lime
import numpy as np
from PIL import Image
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
clean_tensor = transform(img).unsqueeze(0).to(device)

torch.manual_seed(42)
noise = torch.randn_like(clean_tensor) * 0.1
noisy_tensor = clean_tensor + noise

ig = IntegratedGradients(model)
gradcam = LayerGradCam(model, model.layer4)
lime = Lime(model)

feature_mask = torch.arange(196).reshape(1, 1, 14, 14).repeat_interleave(16, dim=2).repeat_interleave(16, dim=3).to(device)

print("Extracting Saliency Maps for Clean and Noisy images... (This takes about 15 seconds due to LIME)")

ig_clean, _ = ig.attribute(clean_tensor, target=1, return_convergence_delta=True)
ig_noisy, _ = ig.attribute(noisy_tensor, target=1, return_convergence_delta=True)

gc_clean = gradcam.attribute(clean_tensor, target=1)
gc_noisy = gradcam.attribute(noisy_tensor, target=1)

lime_clean = lime.attribute(clean_tensor, target=1, feature_mask=feature_mask, n_samples=200)
lime_noisy = lime.attribute(noisy_tensor, target=1, feature_mask=feature_mask, n_samples=200)

def get_sim(attr1, attr2):
    return F.cosine_similarity(attr1.flatten().unsqueeze(0), attr2.flatten().unsqueeze(0)).item()

ig_sim = get_sim(ig_clean, ig_noisy)
gc_sim = get_sim(gc_clean, gc_noisy)
lime_sim = get_sim(lime_clean, lime_noisy)

print("\n=== PHASE 2: ROBUSTNESS METRICS ===")
print(f"Integrated Gradients Similarity: {ig_sim:.4f}")
print(f"Grad-CAM Similarity:           {gc_sim:.4f}")
print(f"LIME Similarity:               {lime_sim:.4f}")
print("===================================\n")

def process_attr(attr):
    arr = attr.squeeze().cpu().detach().numpy()
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    arr = np.maximum(arr, 0)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    return arr

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0,0].imshow(img.resize((224,224)))
axes[0,0].set_title("Clean Image")

axes[0,1].imshow(process_attr(ig_clean), cmap='magma')
axes[0,1].set_title("IG (Clean)")

axes[0,2].imshow(process_attr(gc_clean), cmap='jet')
axes[0,2].set_title("Grad-CAM (Clean)")

axes[0,3].imshow(process_attr(lime_clean), cmap='jet')
axes[0,3].set_title("LIME (Clean)")

noisy_img_display = noisy_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
noisy_img_display = np.clip((noisy_img_display * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

axes[1,0].imshow(noisy_img_display)
axes[1,0].set_title("Noisy Image (+10% Gaussian)")

axes[1,1].imshow(process_attr(ig_noisy), cmap='magma')
axes[1,1].set_title(f"IG (Sim: {ig_sim:.2f})")

axes[1,2].imshow(process_attr(gc_noisy), cmap='jet')
axes[1,2].set_title(f"Grad-CAM (Sim: {gc_sim:.2f})")

axes[1,3].imshow(process_attr(lime_noisy), cmap='jet')
axes[1,3].set_title(f"LIME (Sim: {lime_sim:.2f})")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.savefig("robustness_comparison.png", bbox_inches='tight', dpi=300)

print("✅ SUCCESS: Saved visual comparison as 'robustness_comparison.png'")