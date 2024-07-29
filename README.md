\usepackage{hyperref}
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
\href{https://www.dropbox.com/scl/fi/owhe3zy6rouup9x0biocs/best.pt?rlkey=czklru2w1lsvrdtjgm790rh2s&st=1h8d5zds&dl=0}{Download best.pt}

