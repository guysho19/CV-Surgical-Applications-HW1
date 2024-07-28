from create_pseudo_labels_id import *


def main():
    model_path = '/home/student/HW1_CV/yolov8n_pseudo_trained.pt'
    video_path= '/datashare/HW1/ood_video_data/4_2_24_A_1.mp4'
    images_output_dir_video1 = '/home/student/HW1_CV/ood_pseudo_data/images'
    labels_output_dir_video1 = '/home/student/HW1_CV/ood_pseudo_data/labels'

    model = load_model(model_path)
    os.makedirs(images_output_dir_video1, exist_ok=True)
    os.makedirs(labels_output_dir_video1, exist_ok=True)
    print(f'Processing video from path = {video_path}')
    process_video(video_path, model, images_output_dir_video1, labels_output_dir_video1,max_confidence_threshold=0.93,average_confidence_threshold=0.91, video_str="video1")



if __name__ == "__main__":
    main()
