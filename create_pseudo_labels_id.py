import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO


def load_model(model_path):
    """Load the YOLO model."""
    model = YOLO(model_path)
    return model

# def preprocess_frame(frame):
#     """Preprocess the frame for YOLO model input."""
#     height, width, _ = frame.shape
#     return frame, height, width

def save_frame_and_labels(frame, yolo_labels, frame_count, images_output_dir, labels_output_dir,video_str="video1"):
    """Save the frame and its corresponding labels."""
    frame_filename = os.path.join(images_output_dir, f'{video_str}_frame_{frame_count}.png')
    label_filename = os.path.join(labels_output_dir, f'{video_str}_frame_{frame_count}.txt')

    cv2.imwrite(frame_filename, frame)

    with open(label_filename, 'w') as label_file:
        for label in yolo_labels:
            formatted_label = [int(label[0])] + label[1:]
            label_file.write(' '.join(map(str, formatted_label)) + '\n')


def process_video(video_path, model, images_output_dir, labels_output_dir, max_confidence_threshold=0.975,
                  average_confidence_threshold=0.965, video_str="video1"):
    """Process a video and save frames and labels."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    last_frame_count_added = -10000
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame, height, width = preprocess_frame(frame)
        results = model(frame, stream=True)

        for result in results:
            # Extracting bounding boxes in normalized format [x_center, y_center, width, height]
            boxes = result.boxes.xywhn.cpu().numpy()  # Get normalized bounding box coordinates
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Extract class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

            if boxes.shape[0] == 0:  # No objects detected
                print(f'No objects detected in frame {frame_count}. Skipping...')
                continue

            yolo_labels = []
            for i, box in enumerate(boxes):
                x_center, y_center, width, height = box[:4]
                class_id = class_ids[i]
                confidence = confidences[i]

                # Convert to YOLO format [class_id, x_center, y_center, width, height]
                yolo_labels.append([class_id, x_center, y_center, width, height])

            if confidences.size > 0:
                avg_confidence = np.mean(confidences)
                max_confidence = np.max(confidences)
            else:
                avg_confidence = 0
                max_confidence = 0

            print(f'avg_confidence = {avg_confidence}, max_confidence = {max_confidence}')

            if avg_confidence >= average_confidence_threshold and max_confidence >= max_confidence_threshold and not (
                    frame_count in range(last_frame_count_added, last_frame_count_added + 5)):
                last_frame_count_added = frame_count
                print(f'Saving frame {frame_count} and its labels -----------------\n')
                save_frame_and_labels(frame, yolo_labels, frame_count, images_output_dir, labels_output_dir, video_str)
        frame_count += 1

    cap.release()


def main():
    model_path = '/home/student/HW1_CV/yolov8n_trained_2.pt' #changed
    video_paths = ['/datashare/HW1/id_video_data/20_2_24_1.mp4', '/datashare/HW1/id_video_data/4_2_24_B_2.mp4']
    images_output_dir_video1 = '/home/student/HW1_CV/pseudo_data_video1/images'
    labels_output_dir_video1 = '/home/student/HW1_CV/pseudo_data_video1/labels'
    images_output_dir_video2 = '/home/student/HW1_CV/pseudo_data_video2/images'
    labels_output_dir_video2 = '/home/student/HW1_CV/pseudo_data_video2/labels'

    model = load_model(model_path)

    os.makedirs(images_output_dir_video1, exist_ok=True)
    os.makedirs(labels_output_dir_video1, exist_ok=True)
    print(f'Processing video from path = {video_paths[0]}')
    process_video(video_paths[0], model, images_output_dir_video1, labels_output_dir_video1,max_confidence_threshold=0.97,average_confidence_threshold=0.96, video_str="video1")

    os.makedirs(images_output_dir_video2, exist_ok=True)
    os.makedirs(labels_output_dir_video2, exist_ok=True)
    print(f'Processing video from path = {video_paths[1]}')
    #needs lower thresholds
    process_video(video_paths[1], model, images_output_dir_video2, labels_output_dir_video2,max_confidence_threshold=0.96,average_confidence_threshold=0.95, video_str="video2")



if __name__ == "__main__":
    main()
