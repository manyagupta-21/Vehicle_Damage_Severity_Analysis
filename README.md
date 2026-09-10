# Vehicle Damage Severity Analysis

A multi-label deep learning project for detecting and classifying motor vehicle damage severity from images. This repository contains an end-to-end pipeline — preprocessing, exploratory data analysis, and model benchmarking — built on the CarDD (Car Damage Detection) dataset.

---

## 🚗 Project Overview

This project uses computer vision and deep learning to automatically detect and classify six types of vehicle damage from photographs, framed as a **multi-label classification** problem since a single image can contain multiple damage types simultaneously.

### Damage Categories

| # | Category | Description |
|---|---|---|
| 1 | **Dent** | Impact-related deformations |
| 2 | **Scratch** | Surface-level abrasions |
| 3 | **Crack** | Structural breaks and fractures |
| 4 | **Glass Shatter** | Broken windows and glass |
| 5 | **Lamp Broken** | Damaged lights and lamps |
| 6 | **Tire Flat** | Deflated or damaged tires |

---

## 📊 Dataset

The project uses the **CarDD (Car Damage Detection)** dataset in COCO annotation format:

- **Training set:** 2,816 annotated images
- **Multi-label images:** ~38% of images (1,084 of 2,816) contain more than one damage type
- **Class distribution:**
  - Scratch: 1,507 instances
  - Dent: 1,242 instances
  - Lamp Broken: 489 instances
  - Glass Shatter: 469 instances
  - Crack: 434 instances
  - Tire Flat: 219 instances

---

## 🏆 Results

Three architectures were benchmarked on the same multi-label setup (`BCEWithLogitsLoss`, macro-averaged precision/recall/F1):

| Model | Approach | Epochs | Best Macro Precision | Best Macro Recall | **Best Macro F1** |
|---|---|---|---|---|---|
| Custom CNN (Adam) | Trained from scratch | 20 | 0.607 | 0.436 | 0.471 |
| Custom CNN (NAdam, extended) | Trained from scratch | 45 | 0.738 | 0.515 | 0.581 |
| ResNet50 | Transfer learning (frozen backbone, fine-tuned head) | 10 | 0.806 | 0.643 | 0.696 |
| **VGG16** | **Transfer learning (frozen backbone, fine-tuned head)** | **10** | **0.746** | **0.715** | **0.722 (best)** |

**VGG16 was the best-performing model**, striking the best balance between precision and recall among all architectures tested. Both transfer-learning models substantially outperformed the CNN trained from scratch, reflecting the benefit of ImageNet-pretrained features on a relatively small (2,816-image), imbalanced dataset.

---

## ⚙️ Technologies & Dependencies

**Core libraries:** PyTorch, TorchVision, NumPy, Pandas, Pillow (PIL), Matplotlib
**Evaluation:** scikit-learn (`precision_score`, `recall_score`, `f1_score`, macro-averaged)

---

## 🧠 Modelling Approach

- **Loss function:** `BCEWithLogitsLoss` (standard for multi-label classification, one independent binary decision per class)
- **Optimizers:** Adam (baseline CNN, VGG16, ResNet50 heads), NAdam (extended baseline CNN training)
- **Transfer learning setup:** For VGG16 and ResNet50, the pretrained (ImageNet-1K) convolutional backbone is frozen, and only the final classification layer is fine-tuned on the CarDD data
- **Data augmentation:** Random horizontal flipping, rotation (±10°), random resized cropping, color jitter, and ImageNet-standard normalization
- **Thresholding:** Sigmoid outputs converted to binary predictions at a 0.5 threshold per class
- **Best-model checkpointing:** Model weights saved whenever validation loss (or F1, depending on run) improved

---

## 📁 Repository Structure

```
Vehicle_Damage_Severity_Analysis/
├── data_preprocessing.ipynb      # Raw dataset loading, COCO annotation parsing, validation
├── image_EDA.ipynb               # Exploratory data analysis, class distributions, visualizations
├── Modelling.ipynb               # CarDDDataset class, training/validation loops, CNN/VGG16/ResNet50 benchmarking
└── README.md
```

### Notebook Descriptions

- **`data_preprocessing.ipynb`**: Loads raw images and COCO-format annotations, validates the dataset, and prepares it for modelling.
- **`image_EDA.ipynb`**: Visual exploration of damage types, class distributions, and image characteristics; surfaces the class imbalance and multi-label overlap patterns.
- **`Modelling.ipynb`**: Defines a custom `CarDDDataset` (PyTorch `Dataset`), builds the augmentation pipeline, trains a baseline CNN from scratch, then fine-tunes VGG16 and ResNet50 via transfer learning, evaluating all three with macro precision/recall/F1.

---

## 🚀 Usage

```bash
pip install torch torchvision numpy pandas pillow matplotlib scikit-learn
```

1. Clone the repository and place the CarDD dataset (COCO format) in the expected directory structure.
2. Run `data_preprocessing.ipynb` to load and validate the data.
3. Run `image_EDA.ipynb` to explore class distributions and image characteristics.
4. Run `Modelling.ipynb` to train the baseline CNN and fine-tune VGG16/ResNet50, and reproduce the benchmark results above.

---

## 🔑 Key Insights

- **Transfer learning wins decisively** on a dataset this size, both VGG16 and ResNet50 outperformed a CNN trained from scratch by a wide margin, even with just 10 fine-tuning epochs.
- **Multi-label overlap is common:** ~38% of images carry more than one damage label, which the modelling pipeline explicitly accounts for via independent per-class binary predictions.
- **Class imbalance remains a challenge:** Tire Flat, the rarest class, likely drags down macro-averaged recall across all models; this is the clearest lever for further improvement.

---

## Author

**Manya Gupta**: [@manyagupta-21](https://github.com/manyagupta-21)

## Dataset Citation

Based on the CarDD (Car Damage Detection) dataset, COCO annotation format.

---

**Note:** Ensure the CarDD dataset is placed at the correct path before running the notebooks.
