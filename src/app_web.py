import cv2
import numpy as np
import onnxruntime as ort
import time
from flask import Flask, render_template, Response, jsonify

global_face_detected = False

app = Flask(__name__)

# 18개 피부 지표 이름
feature_names = [
    '여드름 심각도', '블랙헤드', '화이트헤드', '모공 크기', '유분기', 
    '피부 자극', '피부 민감도', '붉은기', '눈가 잔주름', '눈가 부기', 
    '다크서클', '이마 주름', '피부 탄력 저하', '수분 부족', 
    '다크스팟(기미)', '여드름 흉터', '불균형한 피부톤', '주근깨'
]

print("모델을 불러오는 중...")
# 모델 파일 연결
session = ort.InferenceSession("models/skin_model.onnx")
input_name = session.get_inputs()[0].name
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 결과를 저장할 전역 변수
latest_result = {
    "status": "분석중",
    "scores": {}
}

def preprocess(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def analyze_skin(face_roi):
    input_data = preprocess(face_roi)
    outputs = session.run(None, {input_name: input_data})
    
    # 18개 지표 점수를 바로 가져옴
    scores = outputs[0][0]
    
    # 0,  5 사이로 보정 clip?ping
    scores = np.clip(scores, 0.0, 5.0)
    
    # 딕셔너리 생성
    result_dict = {name: round(float(score), 1) for name, score in zip(feature_names, scores)}
    return result_dict

def generate_frames():
    global latest_result
    global global_face_detected 
    cap = cv2.VideoCapture(0) # 카메라가 안 켜지면 1로 변경
    last_analysis_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(150, 150))
            
            if len(faces) > 0:
                global_face_detected = True
            else:
                global_face_detected = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = frame[y:y+h, x:x+w]
                
                current_time = time.time()
                if current_time - last_analysis_time > 3:
                    # 분석 실행
                    analyzed_scores = analyze_skin(face_roi)
                    
                    latest_result["status"] = "Complete"
                    latest_result["scores"] = analyzed_scores
                    last_analysis_time = current_time
                    
                # 화면에는 심플하게 분석 중임을 표시
                cv2.putText(frame, "AI Scannnnnnnning", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # HTML 파일을 연결
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_result')
def get_result():
    return jsonify(latest_result)

@app.route('/face_status')
def face_status():
    return jsonify({"detected": global_face_detected})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
