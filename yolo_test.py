import cv2
import numpy as np
import torch
import ultralytics
import supervision as sv
import os
from ultralytics import YOLO


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#학습된 모델 가져오기
model = YOLO("best.pt")

def get_files_in_folder(folder_path):
    file_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_list.append(filename)
    return file_list

folder_path = ' '   # 폴더 경로
files = get_files_in_folder(folder_path)

for file in files:
    file = str(file)
    file = (' ' + file)   # 폴더 경로

    # 영상 가져오기
    cap = cv2.VideoCapture(file)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # 박스 환경설정
    box_annotator = sv.BoxAnnotator( thickness=2,text_thickness=2, text_scale=1 )

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        results = model.predict(img)[0]

        classes = ["fist", "hammer", "knife"]
        detections = sv.Detections.from_yolov8(results)
        high_confidence_detections = detections[detections.confidence >= 0.6 ]     # 임계값 조정
        labels = [ f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in high_confidence_detections]
        img = box_annotator.annotate(scene=img, detections=high_confidence_detections, labels=labels )
        small_img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("YoLO Test", small_img)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
