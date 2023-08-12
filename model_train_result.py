## 이미지 저장 할 필요 x 영상에서 바로 모델 -> 결과 이미지를 저장하는 걸로 변경

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

model = YOLO("/Users/sooox89/Desktop/workspace/pythonProject/computervision/I_CU/best_custom.pt")  # 여기 적용할 모델

#박스 환경설정
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

cnt = 0
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


folder_path = '/Users/sooox89/Desktop/workspace/pythonProject/computervision/I_CU/real_test_video'  # 영상 폴더 지정
files = get_files_in_folder(folder_path)
count_frame= []
for file in files:
    file = str(file)
    output_path = './'+ file +'/ex/ex_mask_background_subtraction_output.mp4'
    file = ("/Users/sooox89/Desktop/workspace/pythonProject/computervision/I_CU/real_test_video/" + file)  # 영상 폴더 지정
    print(file)

    # 영상 불러오기, 출력 영상 설정
    video_path = file

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v') # 출력 비디오 코덱

    # 영상 크기 설정
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # 배경 제거를 위해 객체 생성 및 커널 설정
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False) # detectShadows=False : 그림자 감지 안함
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False) # detectShadows=False : 그림자 감지 안함

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) # 배경 제거 결과에 적용할 모폴로지 연산 커널 설정

    frame_num = 0

    # 창 생성
    cv2.namedWindow('background extraction video', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        results = model.predict(img)[0]

        frame_num += 1

        # 프레임 크기 절반으로 축소
        resized_img = cv2.resize(img, None, fx=0.5, fy=0.5)

        # 배경 제거
        background_extraction_mask = fgbg.apply(resized_img)
        background_extraction_mask = cv2.dilate(background_extraction_mask, kernel, iterations=1) # cv2.dilate : 배경제거 마스크 확장

        # 크기 복원
        background_extraction_mask = cv2.resize(background_extraction_mask, (width, height))

        background_extraction_mask = np.stack((background_extraction_mask,) * 3, axis=-1)
        bitwise_image = cv2.bitwise_and(img, background_extraction_mask)

        # 창 크기 줄이기
        window_width = int(width * 0.7)
        window_height = int(height * 0.7)
        cv2.resizeWindow('background extraction video', window_width, window_height)

        # 배경이 제거된 프레임만 저장
        out.write(bitwise_image)  # 수정: 배경이 제거된 bitwise_image를 저장

        results = model.predict(bitwise_image)[0]

        classes = ["fist", "hammer", "knife"]
        detections = sv.Detections.from_yolov8(results)
        high_confidence_detections = detections[detections.confidence >= 0.5]
        labels = [f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in high_confidence_detections]
        bitwise_image = box_annotator.annotate(scene=bitwise_image, detections=high_confidence_detections, labels=labels)

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

            cv2.imwrite("/Users/sooox89/Desktop/workspace/pythonProject/computervision/I_CU/result/best_custom_back/frame%d.png" % cnt, bitwise_image)  # 여기 앞부분에 결과이미지를 저장할 주소 적어주시면 됩니다.
            print('Saved frame number :', str(int(cap.get(1))//15))
            cnt += 1
    count_frame.append(cnt)



data = {'filename': list_filename,
        'fist': list_fist,
        'hammer': list_hammer,
        'knife': list_knife,
        'conf': list_confidence}

test_result = pd.DataFrame(data)

print(test_result.shape)

test_result.to_csv('/Users/sooox89/Desktop/workspace/pythonProject/computervision/I_CU/result/best_custom_back.csv', header=True, index=True)  # 여기 csv 저장할 파일 경로와 저장명을 지정해주세요
