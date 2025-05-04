import torch
from pathlib import Path
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('./250123_1148_yolo.pt')

# 이미지 폴더 경로 설정
image_folder = Path('./image_user_resize')

# 지원하는 이미지 확장자
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# 폴더 내의 모든 이미지 파일에 대해 예측 수행
for img_path in image_folder.iterdir():
    if img_path.suffix.lower() in image_extensions:
        # 이미지에 대한 예측 수행
        results = model.predict(str(img_path))
        
        for result in results:
            # 예측된 클래스와 confidence 가져오기
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                
                # 클래스 이름 가져오기 (모델의 names 딕셔너리 사용)
                pred_class = result.names[cls]
                
                # 결과 출력
                print(f"{pred_class} -> {img_path.name}, {conf:.2f}")
