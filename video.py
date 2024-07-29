import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO

def load_model(model_path):
    """Load the YOLO model."""
    model = YOLO(model_path)
    return model

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


def draw_boxes_from_results(frame, results):
    """
    Draw bounding boxes with labels and confidence scores on the image.

    Args:
        image_path (str): Path to the input image.
        results (Results): YOLO detection results.

    Returns:
        Annotated image as a numpy array.
    """
    # Load the image
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


def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame) #changed
        frame_with_boxes = draw_boxes_from_results(frame,results[0]) #may change a bit

        out.write(frame_with_boxes)

    cap.release()
    out.release()


def main():
    model_path = '/home/student/HW1_CV/yolov8n_ood_pseudo_trained.pt'
    ood_video_paths = [
        '/datashare/HW1/ood_video_data/surg_1.mp4',
        '/datashare/HW1/ood_video_data/4_2_24_A_1.mp4'
    ]  # surge1 is a short video for checking
    output_video_dirs = [
        '/home/student/HW1_CV/ood_prediction_video1/video',
        '/home/student/HW1_CV/ood_prediction_video2/video'
    ]
    output_video_files = [
        'output_surg_1.mp4',
        'output_4_2_24_A_1.mp4'
    ]

    model = load_model(model_path)

    for input_video, output_dir, output_file in zip(ood_video_paths, output_video_dirs, output_video_files):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        print(f'Processing video from path = {input_video}')
        process_video(input_video, output_path, model)



if __name__ == "__main__":
    main()








