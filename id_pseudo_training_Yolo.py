import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import yaml
from ultralytics import YOLO

#was train 15 and train 152 in runs/detect now in train 17 and 172
print(f'load yolo model')
model = YOLO("yolov8n_trained_2.pt") #changed
results = model.train(data='pseudo_model_data.yaml', epochs=50, imgsz=640, batch=8)# Load the model, no augment
print('pseudo trained model was saved to path = /home/student/HW1_CV/yolov8n_pseudo_trained.pt')
model.save('yolov8n_pseudo_trained_2.pt')

Evaluation = model.val(data='pseudo_model_data.yaml')#changed from final





