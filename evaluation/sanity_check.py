import argparse
import random
import cv2
from torchvision import datasets

from inference.predict_classifier import predict_faces
from inference.detect_faces import detect_faces_bgr


LABEL_NAMES = {0: "real", 1: "fake"}


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check predictions on random samples")
    parser.add_argument("--data-dir", type=str, default="data/dataset/test")
    parser.add_argument("--model", type=str, default="resnet18", choices=["cnn", "resnet18"])
    parser.add_argument("--model-path", type=str, default="resnet_face_classifier.pth")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--balanced", action="store_true", help="sample an equal number from each class")
    parser.add_argument("--face-crop", action="store_true", help="detect and crop face before prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = datasets.ImageFolder(args.data_dir)
    print("class_to_idx:", ds.class_to_idx)

    if len(ds.samples) == 0:
        print("No samples found.")
        return

    indices = []
    if args.balanced:
        indices_by_label = {}
        for i, (_, label_idx) in enumerate(ds.samples):
            indices_by_label.setdefault(label_idx, []).append(i)

        class_labels = sorted(indices_by_label.keys())
        per_class = max(1, args.num_samples // max(1, len(class_labels)))
        extra = args.num_samples - per_class * len(class_labels)

        for label_idx in class_labels:
            choices = indices_by_label[label_idx]
            random.shuffle(choices)
            take = per_class + (1 if extra > 0 else 0)
            extra = max(0, extra - 1)
            indices.extend(choices[:take])
    else:
        indices = list(range(len(ds.samples)))
        random.shuffle(indices)
        indices = indices[:args.num_samples]

    for idx in indices:
        path, orig_label = ds.samples[idx]
        true_label = LABEL_NAMES[orig_label]

        img = cv2.imread(path)
        if img is None:
            print(f"{path}: could not read")
            continue

        if args.face_crop:
            faces = detect_faces_bgr(img)
            if not faces:
                print(f"{path} | true={true_label} | no face detected")
                continue
            x, y, w, h = faces[0]
            img = img[y:y + h, x:x + w]

        pred = predict_faces([img], model_name=args.model, model_path=args.model_path)[0]
        pred_label = pred["label"]
        prob_real = pred["prob_real"]
        prob_fake = pred["prob_fake"]

        print(f"{path} | true={true_label} | pred={pred_label} | p_real={prob_real:.3f} p_fake={prob_fake:.3f}")


if __name__ == "__main__":
    main()
