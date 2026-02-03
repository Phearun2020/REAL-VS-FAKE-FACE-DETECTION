# Technical Report: Real vs Fake Face Detection (AI-Generated vs Real)

## 1. Objective
The goal of this project is to select, design, and implement deep learning models to detect real vs AI-generated faces. The system should train and evaluate at least two models, compare results, and support image/video inference with visualized face bounding boxes.

## 2. Dataset
**Dataset:** 140K Real and Fake Faces (Kaggle)
- Split into train/validation/test sets using a 70/15/15 split.
- Directory layout:

```
data/classification/
├── train/real, train/fake
├── val/real, val/fake
└── test/real, test/fake
```

**Label mapping:** 0 = real, 1 = fake (see `utils/dataset.py`).

## 3. Methods
### 3.1 Model A: CNN Classifier (Baseline)
A custom CNN with three convolutional blocks and two fully connected layers is trained to classify face crops as real or fake.
- Model: `models/cnn_classifier.py`
- Training: `training/train_classifier.py`
- Evaluation: `evaluation/eval_classifier.py`

### 3.2 Model B: ResNet18 Classifier (Transfer Learning)
A ResNet18 backbone is used with the final fully connected layer replaced for binary classification.
- Model: `models/resnet_classifier.py`
- Training: `training/train_resnet_classifier.py`
- Evaluation: `evaluation/eval_resnet_classifier.py`

### 3.3 Detection + Inference Pipeline
For end-to-end detection on images and videos:
- Face detection: Haar cascade (`inference/detect_faces.py`).
- Classification: CNN or ResNet18 (`inference/predict_classifier.py`).
- Visualization: bounding boxes + labels (`inference/visualize.py`).
- Web UI: Flask app (`webapp/app.py`).

## 4. Experimental Setup
- Input size: 224x224 with ImageNet normalization.
- Optimizer: Adam.
- Loss: Cross-entropy.
- ResNet18 run for comparison used 1 epoch (initial baseline).

## 5. Results (Test Set)
| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| CNN | 0.9666 | 0.9760 | 0.9440 | 0.9597 |
| ResNet18 | 0.9754 | 0.9764 | 0.9743 | 0.9754 |

## 6. Discussion
- The CNN baseline achieves strong overall accuracy and balanced precision/recall on the relabeled dataset.
- ResNet18 trained for 10 epochs outperforms the CNN baseline on all metrics.
- Further improvements are possible with ImageNet initialization and additional data augmentation.

## 7. Conclusion
The project meets the requirement of implementing and comparing at least two deep learning models. The CNN provides a strong baseline, while ResNet18 offers a clear path for improvement with longer training. The system also includes a working inference pipeline and a basic web app for image/video detection with bounding box visualization.

## 8. Future Work
- Train ResNet18 for more epochs and explore pretrained weights.
- Replace Haar cascade with a stronger face detector (e.g., YOLO) using generated labels.
- Add video-level temporal aggregation (LSTM/GRU) for sequence modeling.

## 9. References
```
Kaggle Dataset: 140K Real and Fake Faces
Kaggle Dataset: Real vs AI-Generated Faces (philosopher0808)
https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset/data
```
