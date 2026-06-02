import os
import cv2
import shutil

# OpenCV의 기본 얼굴 인식기 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_and_save(input_base_dir, output_base_dir):
    # 출력 폴더가 없으면 만듦
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    processed_count = 0
    failed_count = 0

    # 폴더 내의 모든 사진을 하나씩 탐색
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # 기존 폴더 구조를 새 폴더에도 똑같이 생성
                rel_path = os.path.relpath(root, input_base_dir)
                target_dir = os.path.join(output_base_dir, rel_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                target_path = os.path.join(target_dir, file)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 얼굴 찾기
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

                if len(faces) > 0:
                    # 얼굴을 찾으면 주변 여백을 15% 정도 남기고 크롭
                    x, y, w, h = faces[0]
                    margin = int(max(w, h) * 0.15)
                    y1 = max(0, y - margin)
                    y2 = min(img.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(img.shape[1], x + w + margin)

                    cropped_img = img[y1:y2, x1:x2]
                    cv2.imwrite(target_path, cropped_img)
                    processed_count += 1
                else:
                    # 가려져서 얼굴을 못 찾으면 원본을 그대로 복사 << 데이터 누락 방지
                    shutil.copy(img_path, target_path)
                    failed_count += 1

                if (processed_count + failed_count) % 100 == 0:
                    print(f" 크롭 완료: {processed_count}건 | 얼굴 못찾음: {failed_count}건")

    print(f"✅ {input_base_dir} 완료 (총 {processed_count + failed_count}장 처리)")

print("=== Train Data 얼굴 크롭 시작 ===")
crop_and_save('dataset/train', 'dataset_cropped/train')

print("\nValid Data 얼굴 크롭 시작 ===")
crop_and_save('dataset/valid', 'dataset_cropped/valid')