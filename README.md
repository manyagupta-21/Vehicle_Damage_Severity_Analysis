# Vehicle Damage Severity Analysis

A comprehensive machine learning project for detecting and classifying motor vehicle damage severity using deep learning. This repository contains end-to-end analysis and modeling of car damage from image data.

## Project Overview

This project leverages computer vision and deep learning to automatically detect and classify various types of motor vehicle damage including dents, scratches, cracks, glass damage, lamp damage, and tire issues. The analysis uses the CarDD (Car Damage Detection) dataset with multi-label classification.

### Damage Categories

The model identifies six types of vehicle damage:
1. **Dent** - Impact-related deformations
2. **Scratch** - Surface-level abrasions
3. **Crack** - Structural breaks and fractures
4. **Glass Shatter** - Broken windows and glass
5. **Lamp Broken** - Damaged lights and lamps
6. **Tire Flat** - Deflated or damaged tires

## Repository Structure

```
motor_damage_severity/
├── data_preprocessing.ipynb      # Data cleaning and preparation
├── image_EDA.ipynb              # Exploratory Data Analysis with visualizations
├── Modelling.ipynb              # Deep learning model development and training
└── README.md                    # This file
```

### Notebook Descriptions

- **data_preprocessing.ipynb**: Handles raw dataset loading, annotation parsing, and data validation
- **image_EDA.ipynb**: Visual exploration of the damage dataset, class distributions, and image characteristics
- **Modelling.ipynb**: Multi-label classification model using deep learning (PyTorch), including:
  - COCO annotation parsing
  - Dataset preparation and augmentation
  - Model architecture and training
  - Performance evaluation with precision, recall, and F1-score

## Dataset

The project uses the **CarDD (Car Damage Detection)** dataset organized in COCO format:
- **Training Set**: 2,816 annotated images
- **Validation Set**: Variable sized validation split
- **Test Set**: Holdout test evaluation
- **Multi-label Format**: Images can contain multiple damage types
- **Class Distribution**:
  - Dent: 1,242 instances
  - Scratch: 1,507 instances
  - Crack: 434 instances
  - Glass Shatter: 469 instances
  - Lamp Broken: 489 instances
  - Tire Flat: 219 instances (class imbalance noted)

## Technologies & Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities and pre-trained models
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Pillow (PIL)**: Image processing
- **Matplotlib**: Data visualization

### ML/Evaluation Tools
- **scikit-learn**: Metrics (precision, recall, F1-score)

## Key Features

✅ **Multi-label Classification**: Handles images with multiple damage types  
✅ **Data Augmentation**: Training-time augmentation for better generalization  
✅ **ImageNet Normalization**: Uses standard ImageNet mean/std for transfer learning  
✅ **Comprehensive Evaluation**: Precision, recall, and F1-score metrics  
✅ **COCO Format Support**: Works with industry-standard annotation format  
✅ **Class Imbalance Handling**: Awareness of imbalanced damage type distributions  

## Model Architecture

The modeling notebook implements:
- Custom `CarDDDataset` class for efficient data loading
- Data augmentation pipeline:
  - Random horizontal flipping
  - Rotation (±10°)
  - Random resized cropping
  - Color jitter
  - Normalization with ImageNet statistics
- Multi-label classification head with binary cross-entropy loss
- PyTorch DataLoader for batch processing

## Usage

To use this repository:

1. Clone the repository
2. Ensure you have the CarDD dataset in the correct directory structure
3. Install required dependencies: `pip install torch torchvision numpy pandas pillow matplotlib scikit-learn`
4. Run notebooks in order:
   - Start with `data_preprocessing.ipynb`
   - Follow with `image_EDA.ipynb`
   - Finally run `Modelling.ipynb` for model training

## Model Evaluation

The project evaluates model performance using:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive instances
- **F1-Score**: Harmonic mean of precision and recall

## Key Insights

- **Multi-label Challenge**: ~38% of images contain multiple damage types (1,084 out of 2,816)
- **Class Imbalance**: Tire flat is significantly underrepresented
- **Damage Frequency**: Scratches and dents are most common damage types

## Future Enhancements

Potential improvements for this project:
- Handle class imbalance with weighted loss functions
- Implement focal loss for hard example mining
- Fine-tune transfer learning models
- Add object detection for damage localization
- Deploy as API for real-world usage
- Explore ensemble methods

## Author

**Manya Gupta** - [@manyagupta-21](https://github.com/manyagupta-21)

## License

This project is open source. Please check LICENSE file for details.

## Dataset Citation

Dataset based on the CarDD (Car Damage Detection) dataset in COCO format.

---

**Note**: Ensure proper directory paths to the CarDD dataset before running notebooks.
