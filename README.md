# Garbage Image Classification using MobileNetV2

A deep learning image classification project that automatically classifies household waste into 12 different categories using **Transfer Learning with MobileNetV2**.

The project focuses on building a practical waste classification pipeline, handling class imbalance, applying data augmentation, and analyzing model errors using classification metrics and a confusion matrix.

---

## Project Overview

Automated waste classification can help improve recycling and waste-sorting systems by reducing the need for manual sorting.

In this project, an image classification model was developed to classify waste images into the following 12 categories:

- Battery
- Biological
- Brown Glass
- Cardboard
- Clothes
- Green Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash
- White Glass

The dataset contains **15,515 images** with a highly imbalanced distribution between classes.

The number of images per class ranges from approximately **607 to 5,325 images**.

To address this imbalance, the project uses **class-weighted training** rather than simply ignoring the distribution differences.

---

## Dataset

The project uses the **Garbage Classification** dataset.

The dataset contains images belonging to 12 waste categories.

Because the dataset is relatively large and is not included in this repository, it should be downloaded separately and the dataset path should be updated in the notebook.

---

## Technologies & Libraries

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Pipeline

The project follows the following workflow:

### 1. Exploratory Data Analysis

- Inspected the dataset structure
- Counted images in each class
- Visualized class distribution
- Displayed sample images

### 2. Data Preparation

The dataset was divided into:

- Training set
- Validation set
- Test set

Stratified splitting was used to maintain class distributions across the datasets.

### 3. Handling Class Imbalance

The dataset contains a significant imbalance between classes.

Class weights were calculated using:

```python
compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label_id"]),
    y=train_df["label_id"]
)
```

These weights were then passed to the model during training.

### 4. Data Augmentation

Data augmentation was applied during training to improve model generalization.

The augmentation pipeline includes:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast

### 5. Transfer Learning

The project uses **MobileNetV2 pretrained on ImageNet**.

MobileNetV2 was selected because it provides a good balance between:

- Classification performance
- Computational efficiency
- Model size
- Potential deployment on resource-constrained systems

The original ImageNet classification head was removed and replaced with a custom classification head consisting of:

- Global Average Pooling
- Dropout
- Dense Softmax classification layer

### 6. Two-Phase Training

#### Phase 1 — Feature Extraction

The MobileNetV2 backbone was frozen and only the new classification head was trained.

#### Phase 2 — Fine-Tuning

The last 40 layers of MobileNetV2 were unfrozen and trained using a much smaller learning rate.

This allowed the pretrained features to adapt to the waste classification problem.

---

## Model Architecture

```text
Input Image
    │
    ▼
Data Augmentation
    │
    ▼
MobileNetV2
(pretrained on ImageNet)
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense Layer
(12 classes)
    │
    ▼
Softmax
    │
    ▼
Predicted Waste Class
```

---

## Results

The final model achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **91.97%** |
| Weighted F1 Score | **0.9204** |
| Macro F1 Score | **0.8860** |
| Test Images | **2,328** |
| Misclassified Images | **187** |

The model performed particularly well on:

| Class | F1 Score |
|---|---:|
| Clothes | **0.9823** |
| Shoes | **0.9481** |
| Biological | **0.9444** |

Some of the more challenging classes were:

| Class | F1 Score |
|---|---:|
| Metal | **0.7615** |
| Brown Glass | **0.8199** |
| White Glass | **0.8264** |

---

## Error Analysis

A confusion matrix was used to understand where the model makes mistakes.

The most noticeable confusion patterns included:

- Clothes ↔ Shoes
- Brown Glass ↔ Metal
- Plastic ↔ White Glass
- Plastic ↔ Metal
- Cardboard ↔ Paper

These errors are understandable because several of these categories can have similar visual characteristics.

For example:

- Cardboard and paper can have similar colors and textures.
- Different glass categories can be visually similar depending on lighting.
- Metal and plastic objects can sometimes have similar shapes or reflective surfaces.
- Clothes and shoes can overlap visually when objects are partially visible.

---

## Fine-Tuning Analysis

An interesting result from the experiment was that fine-tuning did not improve validation performance.

The best validation accuracy was:

```text
Phase 1: 92.69%
Phase 2: 92.22%
```

This suggests that the pretrained MobileNetV2 features were already effective for this dataset.

Fine-tuning the last 40 layers introduced additional trainable parameters without providing an improvement in validation accuracy.

Possible future improvements include:

- Fine-tuning fewer layers
- Using a smaller learning rate
- Increasing the number of minority-class images
- Applying more targeted augmentation
- Experimenting with other pretrained architectures

---

## Key Findings

The project demonstrates that transfer learning can achieve strong performance on waste image classification even when the dataset is highly imbalanced.

The main findings were:

1. **MobileNetV2 provided strong baseline performance.**
2. **Class weighting helped address the dataset imbalance.**
3. **Data augmentation improved the robustness of the training pipeline.**
4. **The model achieved 91.97% test accuracy.**
5. **Most errors occurred between visually similar waste categories.**
6. **Fine-tuning did not improve the frozen-backbone baseline in this experiment.**

---

## Future Improvements

Possible improvements include:

- Experimenting with EfficientNet or other modern CNN architectures
- Increasing minority-class data
- Using targeted augmentation for difficult classes
- Hyperparameter optimization
- Testing different image resolutions
- Model quantization for edge deployment
- Deploying the model as a web or mobile application
- Integrating the classifier with a real-time camera-based waste sorting system

---

## Project Structure

```text
garbage-image-classification/
│
├── garbage-image-classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── wrong_predictions.png
│
└── models/
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/garbage-image-classification.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the Garbage Classification dataset separately and place it on your machine.

### 4. Update the dataset path

Inside the notebook, update:

```python
DATASET_DIR = "path/to/garbage_classification"
```

### 5. Run the notebook

Open:

```text
garbage-image-classification.ipynb
```

and run the cells sequentially.

---

## Author

**Amer Wahid**

Computer Science Student | Data Science & AI Enthusiast

---

## License

This project is intended for educational and research purposes.
