import yaml
from ultralytics import YOLO
# Load YOLOv8 model for training
print(f'load yolo model')
model = YOLO("yolov8n.pt")
# # Train the model with the pic_data.yaml in path = path = /home/student/HW1_CV
# results = model.train(data='pic_data.yaml', epochs=8, imgsz=640, batch=4)
# # Save the intial trained model
# print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained.pt')
# model.save('yolov8n_trained.pt')


# Train the model with the pic_data.yaml in path = path = /home/student/HW1_CV
#train8
results = model.train(data='pic_data.yaml', epochs=50, imgsz=640, batch=4, augment=True)
# Save the intial trained model
print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained_2.pt')
model.save('yolov8n_trained_2.pt')

Evaluation = model.val(data='pic_data.yaml')



# #might want to augmet only some of the augmentations
# results = model.train(data='pic_data.yaml', epochs=50, imgsz=640, batch=16, augment=True)
# # Save the intial trained model
# print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained_batch16_epochs50.pt')
# model.save('yolov8n_trained_batch16_epochs50.pt')
# Evaluation = model.val(data='pic_data.yaml')
#
# print(f'load yolo model')
# model2 = YOLO("yolov8n.pt")
# results2 = model2.train(data='pic_data.yaml', epochs=50, imgsz=640, batch=8, augment=True)
# # Save the intial trained model
# print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained_batch8_epochs50.pt')
# model2.save('yolov8n_trained_batch8_epochs50.pt')
# Evaluation2 = model2.val(data='pic_data.yaml')


# print(f'load yolo model')
# model3 = YOLO("yolov8n.pt")
# results3 = model3.train(data='pic_data.yaml', epochs=100, imgsz=640, batch=16, augment=True)
# # Save the intial trained model
# print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained_batch16_epochs100.pt')
# model3.save('yolov8n_trained_batch16_epochs100.pt')
# Evaluation3 = model3.val(data='pic_data.yaml')
#
# print(f'load yolo model')
# model4 = YOLO("yolov8n.pt")
# results4 = model4.train(data='pic_data.yaml', epochs=100, imgsz=640, batch=8, augment=True)
# # Save the intial trained model
# print('intial model was saved to path = /home/student/HW1_CV/yolov8n_trained_batch8_epochs100.pt')
# model4.save('yolov8n_trained_batch8_epochs100.pt')
# Evaluation4 = model4.val(data='pic_data.yaml')


