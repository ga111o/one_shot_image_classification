import os
from pathlib import Path

def create_yolo_labels(dataset_dir):
    """
    dataset_dir 내의 이미지들에 대해 YOLO 형식의 레이블 파일을 생성합니다.
    모든 객체는 이미지를 100% 채우므로 바운딩 박스는 (0,0,1,1)입니다.
    클래스는 'artwork' 하나입니다.
    """
    # 학습과 검증 데이터 디렉토리
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(img_dir):
            continue
        
        # 이미지 파일 확장자
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        # 각 이미지에 대해 레이블 파일 생성
        for img_file in os.listdir(img_dir):
            if any(img_file.lower().endswith(ext) for ext in img_extensions):
                # 레이블 파일 경로 (확장자만 .txt로 변경)
                label_path = os.path.join(img_dir, Path(img_file).stem + '.txt')
                
                # YOLO 형식으로 레이블 작성
                # 형식: <class> <x_center> <y_center> <width> <height>
                # 클래스 ID는 0 (artwork)
                # 전체 이미지를 커버하므로 중심점은 (0.5, 0.5), 크기는 (1.0, 1.0)
                with open(label_path, 'w') as f:
                    f.write('0 0.5 0.5 1.0 1.0\n')

if __name__ == '__main__':
    dataset_dir = 'dataset'  # 데이터셋 루트 디렉토리
    create_yolo_labels(dataset_dir)
