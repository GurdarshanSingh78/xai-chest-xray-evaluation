# CNN Saliency Evaluation: Faithfulness vs. Sanity in Medical Imaging

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Captum](https://img.shields.io/badge/Captum-Interpretability-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

Visual heatmaps in Explainable AI (XAI) can be dangerously misleading, especially in safety-critical domains like medical imaging. This repository contains a rigorous, reproducible experimental pipeline that evaluates the true robustness of popular CNN saliency methods (Grad-CAM and Integrated Gradients). 

Instead of relying on qualitative visual inspections, this study stress-tests a **ResNet-50** model trained on Chest X-Rays (Normal vs. Pneumonia) using strict quantitative evaluation protocols: **Faithfulness Metrics** (Deletion/Insertion AUC) and **Sanity Checks** (Cascading Parameter Randomization).

## Key Findings & Contributions

Our empirical evaluation revealed a critical, counter-intuitive discrepancy:
* **The Faithfulness Trap:** Integrated Gradients drastically outperformed Grad-CAM on pixel-level faithfulness metrics, making it appear superior.
* **The Sanity Check Failure:** Despite high faithfulness scores, Integrated Gradients completely failed the cascading randomization test. It continued to produce coherent lung outlines even after the model's visual processing layers were randomized, indicating it degraded into a simple edge-detector.
* **Robustness of Grad-CAM:** Grad-CAM produced coarser heatmaps and lower AUC scores, but successfully collapsed into noise during randomization, proving it honestly reflects the actual learned features of the CNN.

## Dataset

This project utilizes the standard benchmark **Chest X-Ray Images (Pneumonia)** dataset. 
* **Task:** Binary Image Classification (Normal vs. Pneumonia)
* **Source:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

*Note: The dataset is excluded from version control. Please download it directly and place it in the root `/chest_xray/` directory before running the pipeline.*

## Repository Structure

```text
├── baseline_training.ipynb       # Fine-tunes ResNet-50 on the Chest X-Ray dataset
├── extract_explanations.py       # Hooks into the model via Captum to generate heatmaps
├── evaluate_faithfulness.py      # Calculates Deletion/Insertion AUC curves
├── evaluate_sanity_checks.py     # Executes cascading weight randomization tests
├── 3x3.png                       # Sanity check grid visualization
├── curves.png                    # Faithfulness AUC curves visualization
└── README.md
```

## Methodology

### 1\. Faithfulness Evaluation

We measure whether the pixels highlighted by the explanation method actually drive the model's prediction.

  * **Deletion:** Iteratively mask the most "important" pixels to zero. A rapid drop in model confidence (Lower AUC) indicates high faithfulness.
  * **Insertion:** Start with a baseline (black) image and iteratively introduce the most "important" pixels. A rapid rise in confidence (Higher AUC) indicates high faithfulness.

### 2\. Sanity Checks (Cascading Randomization)

We evaluate if the explanation methods are sensitive to the model's learned weights.

  * We systematically randomize the weights of the ResNet-50, starting from the Fully Connected (`fc`) classifier down to the final convolutional block (`layer4`).
  * If a saliency method produces the same heatmap for a randomized model as it does for a trained model, it fails the sanity check.

## Visual Evidence & Results Summary

### Faithfulness Metrics (The Trap)
![Deletion and Insertion AUC Curves](curves.png)
*Integrated Gradients shows steeper degradation on deletion and faster rise on insertion, falsely suggesting it is the superior method.*

### Sanity Checks (The Reality)
![Cascading Randomization Sanity Check Grid](3x3.png)
*Notice how Integrated Gradients (middle column) remains practically unchanged even after the visual processing layers are completely randomized (bottom row), failing the sanity check.*


### Quantitative Verdict

| Metric | Integrated Gradients | Grad-CAM |
| :--- | :--- | :--- |
| **Deletion AUC (Lower = Better)** | **0.052** | 0.125 |
| **Insertion AUC (Higher = Better)** | **0.778** | 0.585 |
| **Sanity Check (Model Randomization)** | **FAIL** (Acts as edge detector) | **PASS** (Reflects learned weights) |

## Usage & Reproduction

**1. Clone the repository and navigate to the directory:**

```bash
git clone [https://github.com/GurdarshanSingh78/xai-chest-xray-evaluation.git](https://github.com/GurdarshanSingh78/xai-chest-xray-evaluation.git)
cd xai-chest-xray-evaluation
```

**2. Install dependencies:**

```bash
pip install torch torchvision captum scikit-learn matplotlib numpy pillow
```

**3. Execute the pipeline:**
To fully reproduce this study, first generate the base model weights by running the training notebook, then execute the evaluation scripts.

```bash
# 1. Train the model (generates the .pth weight file)
# Run baseline_training.ipynb in Jupyter/Colab

# 2. Generate the comparative heatmaps
python extract_explanations.py

# 3. Calculate the Deletion and Insertion AUC metrics
python evaluate_faithfulness.py

# 4. Run the Cascading Randomization grid
python evaluate_sanity_checks.py
```

-----

*Developed for research and evaluation in Deep Learning architectures.*
