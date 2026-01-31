# Real vs Fake Face Detection (AI-Generated vs Real)

## 1) Objective
Build an AI system to detect **real vs AI-generated faces**. The system should:
- Train and evaluate at least two deep learning models.
- Compare model performance and discuss results.
- Support image/video input and visualize results with face bounding boxes.

This repo implements two image-level classifiers (**CNN** and **ResNet18**) and includes **data preparation for a YOLO-style detector**.

## 2) Dataset
**Primary dataset:** 140K Real and Fake Faces (Kaggle)
- Download and split into train/val/test.
- Expected directory structure:

```
data/classification/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Label mapping:**
- This project remaps labels to **0 = real**, **1 = fake**.
- See `utils/dataset.py` for details.

### YOLO label generation (for detection model)
`utils/generate_yolo_labels.py` converts the classification dataset into YOLO-style labels using a face detector.

```
PYTHONPATH=. python3 utils/generate_yolo_labels.py
```

Output structure:
```
data/detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 3) Deep Learning Models
This project is designed to satisfy the requirement of **at least two models**.

### Model A: CNN Classifier (implemented)
- **Task:** Classify cropped face images as real or fake.
- **Code:**
  - Model: `models/cnn_classifier.py`
  - Training: `training/train_classifier.py`
  - Evaluation: `evaluation/eval_classifier.py`

### Model B: ResNet18 Classifier (implemented)
- **Task:** Classify cropped face images as real or fake using transfer learning.
- **Code:**
  - Model: `models/resnet_classifier.py`
  - Training: `training/train_resnet_classifier.py`
  - Evaluation: `evaluation/eval_resnet_classifier.py`

### Model C: YOLO-Style Face Detector (data prep included)
- **Task:** Detect faces and classify each bounding box as real/fake.
- **Code:**
  - Data preparation: `utils/generate_yolo_labels.py`
- **Note:** Training/inference for YOLO is not included yet. You can train with a YOLO framework (e.g., Ultralytics) using the generated `data/detection/` labels.

### Optional/Extension: Sequence Model (video-level)
- **Idea:** Aggregate frame-level predictions for video inputs (e.g., LSTM/GRU/Temporal CNN).
- **Status:** Not implemented in this repo yet.

## 4) Training & Evaluation (CNN / ResNet18)
Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python matplotlib scikit-learn seaborn flask pillow
```

Train the CNN classifier:
```
PYTHONPATH=. python3 training/train_classifier.py
```

Evaluate the trained model:
```
PYTHONPATH=. python3 evaluation/eval_classifier.py --num-workers 0
```

Results are saved to:
```
results/cnn/
├── metrics.json
└── confusion_matrix.png
```

Train the ResNet18 classifier:
```
PYTHONPATH=. python3 training/train_resnet_classifier.py
```
Optional flags: `--pretrained` (uses ImageNet weights) and `--freeze-backbone` (train only the final layer).

Evaluate the ResNet18 model:
```
PYTHONPATH=. python3 evaluation/eval_resnet_classifier.py --num-workers 0
```

Results are saved to:
```
results/resnet/
├── metrics.json
└── confusion_matrix.png
```

### Evaluation Summary (Test Set)
| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| CNN | 0.9607 | 0.9667 | 0.9543 | 0.9604 |
| ResNet18 | 0.8706 | 0.8226 | 0.9449 | 0.8795 |

Notes:
- CNN results are from `results/cnn/metrics.json`.
- ResNet18 results are from a **1-epoch** training run; train longer to improve performance.

Discussion:
- The CNN baseline performs strongly with high accuracy and balanced precision/recall on the test set.
- The ResNet18 run used only 1 epoch, which explains the lower accuracy; however, recall is high, indicating strong detection of fake faces at the cost of more false positives.
- With more epochs and optional ImageNet initialization (`--pretrained`), ResNet18 should improve and provide a more robust comparison.

## 5) Inference / Web App
Goal: input image/video -> detect faces -> classify real/fake -> draw bounding boxes.

Implemented inference pipeline:
- Face detection: Haar cascade (`inference/detect_faces.py`)
- Classification: CNN or ResNet18 (`inference/predict_classifier.py`)
- Visualization: bounding boxes + labels (`inference/visualize.py`)

Run the web app:
```
PYTHONPATH=. python3 webapp/app.py
```

Then open: http://localhost:5000

### How to Run the Web App (Step-by-step)
1) Make sure you have model weights:
   - `cnn_face_classifier.pth` for CNN
   - `resnet_face_classifier.pth` for ResNet18
2) Start the server:
```
PYTHONPATH=. python3 webapp/app.py
```
3) Open the app in your browser: `http://localhost:5000`
4) Upload an image or video and select the model.
5) The app will draw bounding boxes and label each face as real or fake.

### Usage Notes
- If no faces are detected, the app will show a warning.
- Video processing samples frames (stride=2) to keep it fast.
- Outputs are saved to `webapp/static/outputs/` and displayed in the UI.
- If you see a "Model weights not found" error, confirm the `.pth` files exist in the project root.

### Screenshots
Add your screenshots here after running the web app:
- `assets/webapp_home.png` (home screen)
- `assets/webapp_result_image.png` (image result with boxes)
- `assets/webapp_result_video.png` (video result with boxes)

## 6) Project Structure
```
models/         # CNN model definition
training/       # training scripts
evaluation/     # evaluation scripts
utils/          # dataset utils + YOLO label generator
inference/      # inference pipeline
webapp/         # Flask web application
results/        # evaluation outputs
```

## 7) Notes on Project Requirements
To fully satisfy the project requirements, ensure you:
- Train **at least two** deep learning models.
- Compare their performance (accuracy/F1/confusion matrix).
- Discuss results in your technical report (max 4 pages).

This repo now includes two models and a basic web app for detection and visualization.
