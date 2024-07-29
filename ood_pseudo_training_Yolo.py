import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO

#train 22 and 222 in runs/detect
print(f'load yolo model')
model = YOLO("yolov8n_pseudo_trained_2.pt")
results = model.train(data='final_model_data.yaml', epochs=50, imgsz=640, batch=8)# Load the model, no augement
print('pseudo trained model was saved to path = /home/student/HW1_CV/yolov8n_ood_pseudo_trained.pt')
model.save('yolov8n_ood_pseudo_trained.pt')

Evaluation = model.val(data='final_model_data.yaml')#change