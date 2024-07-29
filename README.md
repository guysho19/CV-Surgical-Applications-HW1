# CV-Surgical-Applications-HW1
HW1 in the course of Computer Vision Surgical Applications
Order of files running:
1. initial_trained_Yolo.py
2. create_pseudo_labels_id.py
3. id_pseudo_training_Yolo.py
4. create_pseudo_labels_ood.py
5. ood_pseudo_training_Yolo.py
6. run predict.py or video.py

In order to run the files you should be at the path '/home/student/HW1_CV' in the VM 'cvsa2023s-0010'
I also provided my yaml files in the directory yamls. You can access them akso through the VM in path '/home/student/HW1_CV' with the corresponding yaml file name.
In addition the picture for predict.py is from the validation set given in the VM in path: '/datashare/HW1/labeled_image_data/images/val/e398aed5-frame_2832.jpg'
The data for the video.py prediction is in the VM in Paths '/datashare/HW1/ood_video_data/surg_1.mp4', '/datashare/HW1/ood_video_data/4_2_24_A_1.mp4'

## Setup Instructions
1. **Navigate to Project Directory:**
   If you're not already in the project directory, navigate to it:
   ```bash
   cd /home/student/HW1_CV
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run files according to the order above**
   ```bash
   #Run the initial training script for YOLO
   python /home/student/HW1_CV/initial_trained_Yolo.py
   
   #Create pseudo labels for ID data
   python /home/student/HW1_CV/create_pseudo_labels_id.py
   
   #Train YOLO with pseudo labels for ID data
   python /home/student/HW1_CV/id_pseudo_training_Yolo.py
   
   #Create pseudo labels for OOD data
   python /home/student/HW1_CV/create_pseudo_labels_ood.py
   
   #Train YOLO with pseudo labels for OOD data
   python /home/student/HW1_CV/ood_pseudo_training_Yolo.py
   
   #Run predictions (use either predict.py or video.py)
   python /home/student/HW1_CV/predict.py
   #or
   python /home/student/HW1_CV/video.py

## Model weights link:
[Download best.pt](https://www.dropbox.com/scl/fo/af77n4ey2439dxcb6soj2/AFi6G_BsS3gXGcC-7ZTdALQ?rlkey=3lec3dc5n1yb5zhbs0bxyhfnf&st=ehb2kwdg&dl=0)

