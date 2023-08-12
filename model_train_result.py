import cv2
import numpy as np
import torch
import ultralytics
import supervision as sv
import os
from ultralytics import YOLO
import pandas as pd
from pandas import DataFrame, Series

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO("best.pt")  # 여기 적용할 모델

cnt = 0

# class에 맞게 수정
list_filename = list()
list_classname = list()
list_confidence = list()
list_fist = list()
list_hammer = list()
list_knife = list()


def get_files_in_folder(folder_path):
    file_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_list.append(filename)
    return file_list


folder_path = ''  # 영상 폴더 지정
files = get_files_in_folder(folder_path)
count_frame= []
for file in files:
    file = str(file)
    print(file)
    file = ('' + file)  # 영상 폴더 지정

    cap = cv2.VideoCapture(file)

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        results = model.predict(img)[0]

        detections = sv.Detections.from_yolov8(results)
        high_confidence_detections = detections[detections.confidence > 0.5]

        img = box_annotator.annotate(scene=img, detections=high_confidence_detections)

        fist = 0
        hammer = 0
        knife = 0

        if (int(cap.get(1)) % 15 == 0):  # 앞서 불러온 fps 값을 사용하여 0.5초마다 추출 (fps 30일거라 15로 걍 했으니 안 건드셔도 됩니다)

            if high_confidence_detections.class_id.any():

                for i in high_confidence_detections.class_id:
                    if i == 0:
                        fist += 1
                    if i == 1:
                        hammer += 1
                    if i == 2:
                        knife += 1

            list_fist.append(fist)
            list_hammer.append(hammer)
            list_knife.append(knife)

            # list_classname.append(str(high_confidence_detections.class_id)[1:-1])
            list_filename.append(str("frame%d" % cnt))
            list_confidence.append(str(high_confidence_detections.confidence)[1:-1])

            cv2.imwrite(""+"/frame%d.png" % cnt, img)  # 앞부분에 결과이미지를 저장할 주소 적어주시면 됩니다.
            print('Saved frame number :', str(int(cap.get(1))//15))
            cnt += 1
    count_frame.append(cnt)

# class에 맞게 수정
data = {'filename': list_filename, 'fist': list_fist,'hammer': list_hammer,'knife': list_knife, 'conf': list_confidence}

test_result = pd.DataFrame(data)

print(test_result.shape)
print('hammer:',count_frame[1],'knife:',count_frame[3],'buying:',count_frame[11],'fist:',count_frame[13])

test_result.to_csv('best.csv', header=True, index=True)  # 여기 csv 저장할 파일 경로와 저장명을 지정해주세요
