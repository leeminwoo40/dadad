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
``` bash
### 2. yolov8로 이미지 분석하기
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
