import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution

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
image = Image.open(os.path.join(img_path, os.listdir(img_path)[0])).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device).requires_grad_()

ig = IntegratedGradients(model)
layer_gc = LayerGradCam(model, model.layer4[-1])
ig_attr = ig.attribute(input_tensor, target=1, n_steps=50)
gc_attr = LayerAttribution.interpolate(layer_gc.attribute(input_tensor, target=1), input_tensor.shape[2:])

def calc_faithfulness(attr_map, steps=50):
    spatial_attr = torch.sum(attr_map, dim=1).squeeze().cpu().detach().numpy() if attr_map.dim() == 4 and attr_map.shape[1] == 3 else attr_map.squeeze().cpu().detach().numpy()
    h, w = spatial_attr.shape; total_pixels = h * w
    flat_indices = np.argsort(spatial_attr.flatten())[::-1]
    
    del_scores, ins_scores = [], []
    baseline = torch.zeros_like(input_tensor)
    step_size = total_pixels // steps
    
    for i in range(steps + 1):
        num_pixels = min(i * step_size, total_pixels)
        mask = np.zeros(total_pixels)
        mask[flat_indices[:num_pixels]] = 1.0
        mask_tensor = torch.tensor(mask.reshape((1, 1, h, w)), dtype=torch.float32, device=device)
        
        del_img = input_tensor * (1 - mask_tensor)
        ins_img = baseline * (1 - mask_tensor) + input_tensor * mask_tensor
        
        with torch.no_grad():
            del_scores.append(torch.softmax(model(del_img), dim=1)[0, 1].item())
            ins_scores.append(torch.softmax(model(ins_img), dim=1)[0, 1].item())
            
    x_axis = np.linspace(0, 1, steps + 1)
    return x_axis, del_scores, ins_scores, auc(x_axis, del_scores), auc(x_axis, ins_scores)

x_ig, del_ig, ins_ig, d_auc_ig, i_auc_ig = calc_faithfulness(ig_attr)
x_gc, del_gc, ins_gc, d_auc_gc, i_auc_gc = calc_faithfulness(gc_attr)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(x_ig, del_ig, label=f'IG (AUC: {d_auc_ig:.3f})'); axs[0].plot(x_gc, del_gc, label=f'Grad-CAM (AUC: {d_auc_gc:.3f})')
axs[0].set_title('Deletion (Lower is Better)'); axs[0].legend()
axs[1].plot(x_ig, ins_ig, label=f'IG (AUC: {i_auc_ig:.3f})'); axs[1].plot(x_gc, ins_gc, label=f'Grad-CAM (AUC: {i_auc_gc:.3f})')
axs[1].set_title('Insertion (Higher is Better)'); axs[1].legend()

plt.savefig('faithfulness_metrics.png', dpi=300)
print("Saved faithfulness_metrics.png")