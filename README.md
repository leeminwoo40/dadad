
###  yolov8로 이미지 & 영상 분석하기
``` bash
!pip install ultralytics

from ultralytics import YOLO
from IPython.display import display, Image

# 모델 로드
model = YOLO('/content/yolov8n.pt')  # 모델 경로를 적절히 수정해주세요.

# 이미지 로드
img_path = '/content/1.jpg.heic'  # 이미지 경로를 적절히 수정해주세요.

# 이미지 인식 수행
results = model(img_path)

# 결과를 이미지로 저장
results[0].save('/content/output.jpg') 

# 결과 이미지를 Colab에서 보여주기
display(Image('/content/output.jpg'))

# 객체 수 세기
num_objects = len(results[0].boxes)
print(f"이미지에서 {num_objects}개의 개체가 감지되었습니다.")
```
```bash
from google.colab import files
uploaded = files.upload()  # 여기서 '분류영상.mp4'와 'yolov8x.pt'를 업로드

import cv2
from ultralytics import YOLO
from google.colab import files

# 모델 로드
model = YOLO('/content/yolov8x.pt')

# 영상 파일 경로
video_path = '/content/분류영상.mp4'

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 영상의 프레임 크기와 FPS 정보 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 결과를 저장할 비디오 파일 설정
output_path = '/content/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec 사용
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 영상 프레임을 하나씩 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 객체 감지
    results = model(frame)

    # 인식 결과를 이미지에 표시
    annotated_frame = results.plot()  # 결과를 프레임에 오버레이
 
    # 결과 프레임을 output.mp4에 기록
    out.write(annotated_frame)

# 자원 해제
cap.release()
out.release()

# output.mp4를 Google Drive에 저장
from google.colab import drive
drive.mount('/content/drive')
!cp /content/output.mp4 /content/drive/MyDrive/

# Google Drive에서 파일 다운로드
files.download('/content/drive/MyDrive/output.mp4')
```
### 3. yolo 8,11 성능 비교하기

![yolov8](https://github.com/user-attachments/assets/f8c09742-ba38-4096-910e-aa1869919657)

![그래프10](https://github.com/user-attachments/assets/e74c7c49-10b3-4243-acf9-12993ce07778)

![yolov11](https://github.com/user-attachments/assets/0cca4dc2-9ded-4d4b-bd4c-4f4a90b6c8e3)

![그래프2](https://github.com/user-attachments/assets/43c60b77-3ec3-4eb5-ba8d-5f8edbe78c96)







