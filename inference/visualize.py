import cv2


def draw_boxes(image_bgr, boxes, predictions):
    """
    Draw bounding boxes and labels on image.
    predictions: list of dicts from predict_faces
    """
    output = image_bgr.copy()

    for (x, y, w, h), pred in zip(boxes, predictions):
        label = pred.get("label", "unknown")
        prob_real = pred.get("prob_real", 0.0)
        prob_fake = pred.get("prob_fake", 0.0)

        if label == "real":
            color = (0, 200, 0)  # green
            conf = prob_real
        else:
            color = (0, 0, 200)  # red
            conf = prob_fake

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(output, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output
