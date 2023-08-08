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

# 박스 환경설정
box_annotator = sv.BoxAnnotator( thickness=2,text_thickness=2, text_scale=1 )
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
    output_path = './'+ file +'/ex/ex_mask_background_subtraction_output.mp4'
    file = (" " + file)    # 폴더 경로

    # 영상 불러오기, 출력 영상 설정
    video_path = file

    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
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

    # 비디오 프레임 처리
    while True:
        ret, frame = video.read()
        if not ret:
            print('비디오가 끝났거나 오류가 있습니다')
            break

        frame_num += 1

        # 프레임 크기 절반으로 축소
        resized_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        # 배경 제거
        background_extraction_mask = fgbg.apply(resized_frame)
        background_extraction_mask = cv2.dilate(background_extraction_mask, kernel, iterations=1) # cv2.dilate : 배경제거 마스크 확장

        # 크기 복원
        background_extraction_mask = cv2.resize(background_extraction_mask, (width, height))

        background_extraction_mask = np.stack((background_extraction_mask,) * 3, axis=-1)
        bitwise_image = cv2.bitwise_and(frame, background_extraction_mask)

        # 창 크기 줄이기
        window_width = int(width * 0.7)
        window_height = int(height * 0.7)
        cv2.resizeWindow('background extraction video', window_width, window_height)

        # 배경이 제거된 프레임만 저장
        out.write(bitwise_image)  # 수정: 배경이 제거된 bitwise_image를 저장

        results = model.predict(bitwise_image)[0]

        classes = ["fist", "hammer", "knife"]
        detections = sv.Detections.from_yolov8(results)
        high_confidence_detections = detections[detections.confidence >= 0.6]  # 임계값 조정
        labels = [f"{classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in high_confidence_detections]
        bitwise_image = box_annotator.annotate(scene=bitwise_image, detections=high_confidence_detections, labels=labels)

        # 결과 영상 출력
        cv2.imshow('background extraction video', bitwise_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 영상 처리 종료
    video.release()
    out.release()
    cv2.destroyAllWindows()
















