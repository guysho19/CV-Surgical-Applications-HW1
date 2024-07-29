import cv2
import numpy as np
from ultralytics import YOLO


def generate_color_map(num_classes):
    """
    Generate a color map for each class ID.

    Args:
        num_classes (int): Number of classes.

    Returns:
        List of colors for each class ID.
    """
    np.random.seed(42)  # For reproducible colors
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]


def draw_boxes_from_results(image_path, results):
    """
    Draw bounding boxes with labels and confidence scores on the image.

    Args:
        image_path (str): Path to the input image.
        results (Results): YOLO detection results.

    Returns:
        Annotated image as a numpy array.
    """
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}.")

    # Get the bounding boxes, class names, and confidences
    boxes = results.boxes
    names = results.names
    num_classes = len(names)
    colors = generate_color_map(num_classes)  # Generate colors for each class

    # Set font scale and thickness for the text
    font_scale = 2  # Increase font scale for larger text
    font_thickness = 3  # Increase thickness for better visibility

    # Iterate over the detected objects
    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0]
        cls_id = int(box.cls[0])

        # Draw the bounding box with class-specific color
        color = colors[cls_id]
        thickness = 2  # Thickness of the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw the label and confidence score with increased font scale and thickness
        label = f"{names[cls_id]}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text size
        label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_width, text_height = label_size
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10

        # Draw background rectangle for the text
        background_color = (255, 255, 255)  # White background for text
        background_thickness = -1  # Fill the rectangle
        cv2.rectangle(frame, (x1, label_y - text_height - 10), (x1 + text_width, label_y + 10), background_color,
                      background_thickness)

        # Draw the text
        cv2.putText(frame, label, (x1, label_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    return frame


# Load the YOLO model
model_path = '/home/student/HW1_CV/yolov8n_ood_pseudo_trained.pt'
model = YOLO(model_path)
#model.val(data='pic_data.yaml')
# Path to the image
#one_val_pic_path = '/datashare/HW1/labeled_image_data/images/val/e398aed5-frame_2832.jpg'
def process_and_save_image(frame_name):
    # Define paths
    pic_path = f'/datashare/HW1/labeled_image_data/images/val/{frame_name}.jpg'
    save_path = f'/home/student/HW1_CV/{frame_name}.jpg'

    # Predict on the image
    results_list = model(pic_path)
    results = results_list[0]  # Get the first result

    # Draw bounding boxes and save the example image
    annotated_frame = draw_boxes_from_results(pic_path, results)

    # Save the annotated image
    cv2.imwrite(save_path, annotated_frame)

    print(f"Annotated image saved to {save_path}")


# Example usage from validation data from vm

process_and_save_image('e398aed5-frame_2832')



