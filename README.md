## 생성형 인공지능을 활용한 이미지 분류 프로젝트
#####  마크다운 방법
###      구리고등학교
##  1학년
![손흥민](https://github.com/user-attachments/assets/a19c93d9-910f-464b-8cd6-88dce8072019)

``` bash
dkakdakdasdkilsadkljsalkdaslk
```
##### 1. 구구단 만들기
```bash
# 구구단 출력 함수
def print_gugudan():
    for i in range(1, 10):
        for j in range(1, 10):
            print(f"{i} x {j} = {i*j}", end="\t")
        print()

# 구구단 출력
print_gugudan()
```
### 2. yolov8로 이미지 & 영상 분석하기
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
### 3. yolo 8,10,11 성능 비교하기기

![yolov8](https://github.com/user-attachments/assets/f8c09742-ba38-4096-910e-aa1869919657)

![yolov10](https://github.com/user-attachments/assets/c3a5e268-b514-46fc-8461-ca5591c22194)

![yolov11](https://github.com/user-attachments/assets/0cca4dc2-9ded-4d4b-bd4c-4f4a90b6c8e3)








