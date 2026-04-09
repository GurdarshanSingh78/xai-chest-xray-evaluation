import torch
import torch.nn as nn
import copy
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_orig = models.resnet50(pretrained=False)
model_orig.fc = nn.Linear(model_orig.fc.in_features, 2)
model_orig.load_state_dict(torch.load('resnet50_pneumonia_baseline.pth', map_location=device))
model_orig = model_orig.to(device).eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img_path = os.path.join('chest_xray', 'train', 'PNEUMONIA')
image = Image.open(os.path.join(img_path, os.listdir(img_path)[0])).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device).requires_grad_()
original_image = np.clip(np.transpose((input_tensor.squeeze().cpu().detach().numpy() * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + np.array([0.485, 0.456, 0.406]).reshape(3,1,1)), (1,2,0)), 0, 1)

def randomize_weights(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight) if len(layer.weight.shape) >= 2 else nn.init.uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None: nn.init.zeros_(layer.bias)

model_rand_fc = copy.deepcopy(model_orig)
randomize_weights(model_rand_fc.fc)

model_rand_l4 = copy.deepcopy(model_rand_fc)
for module in model_rand_l4.layer4.modules(): randomize_weights(module)

def get_hm(m):
    ig = IntegratedGradients(m); lgc = LayerGradCam(m, m.layer4[-1])
    return np.transpose(ig.attribute(input_tensor, target=1, n_steps=50).squeeze().cpu().detach().numpy(), (1,2,0)), np.expand_dims(LayerAttribution.interpolate(lgc.attribute(input_tensor, target=1), input_tensor.shape[2:]).squeeze().cpu().detach().numpy(), axis=2)

ig_o, gc_o = get_hm(model_orig)
ig_f, gc_f = get_hm(model_rand_fc)
ig_l, gc_l = get_hm(model_rand_l4)

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3): axs[i, 0].imshow(original_image); axs[i, 0].axis('off')
axs[0, 0].set_title("Original")

viz.visualize_image_attr(ig_o, original_image, sign="positive", title="IG (Normal)", plt_fig_axis=(fig, axs[0, 1]), use_pyplot=False)
viz.visualize_image_attr(gc_o, original_image, sign="positive", title="GC (Normal)", plt_fig_axis=(fig, axs[0, 2]), use_pyplot=False)
viz.visualize_image_attr(ig_f, original_image, sign="positive", title="IG (Rand FC)", plt_fig_axis=(fig, axs[1, 1]), use_pyplot=False)
viz.visualize_image_attr(gc_f, original_image, sign="positive", title="GC (Rand FC)", plt_fig_axis=(fig, axs[1, 2]), use_pyplot=False)
viz.visualize_image_attr(ig_l, original_image, sign="positive", title="IG (Rand L4)", plt_fig_axis=(fig, axs[2, 1]), use_pyplot=False)
viz.visualize_image_attr(gc_l, original_image, sign="positive", title="GC (Rand L4)", plt_fig_axis=(fig, axs[2, 2]), use_pyplot=False)

plt.tight_layout()
plt.savefig('sanity_checks_grid.png', dpi=300)
print("Saved sanity_checks_grid.png")