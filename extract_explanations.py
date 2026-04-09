import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet50_pneumonia_baseline.pth', map_location=device))
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = os.path.join('chest_xray', 'train', 'PNEUMONIA')
sample_image_name = os.listdir(img_path)[0]
image = Image.open(os.path.join(img_path, sample_image_name)).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device).requires_grad_()

ig = IntegratedGradients(model)
layer_gc = LayerGradCam(model, model.layer4[-1])
ig_attr = ig.attribute(input_tensor, target=1, n_steps=50)
gc_attr = layer_gc.attribute(input_tensor, target=1)
upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, input_tensor.shape[2:])

original_image = np.clip(np.transpose((input_tensor.squeeze().cpu().detach().numpy() * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + np.array([0.485, 0.456, 0.406]).reshape(3,1,1)), (1,2,0)), 0, 1)
ig_attr_np = np.transpose(ig_attr.squeeze().cpu().detach().numpy(), (1,2,0))
gc_attr_np = np.expand_dims(upsampled_gc_attr.squeeze().cpu().detach().numpy(), axis=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_image); axs[0].set_title("Original X-Ray"); axs[0].axis('off')
viz.visualize_image_attr(ig_attr_np, original_image, method="blended_heat_map", sign="positive", title="Integrated Gradients", plt_fig_axis=(fig, axs[1]), use_pyplot=False)
viz.visualize_image_attr(gc_attr_np, original_image, method="blended_heat_map", sign="positive", title="Grad-CAM", plt_fig_axis=(fig, axs[2]), use_pyplot=False)

plt.tight_layout()
plt.savefig('visual_explanations.png', dpi=300)
print("Saved visual_explanations.png")