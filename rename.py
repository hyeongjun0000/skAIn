import onnx

print("원래 모델을 불러오는 중...")
# 1. 기존의 긴 이름 모델을 읽어옵니다.
model = onnx.load("skin_pro_model_max_acc.onnx")

print("새로운 이름으로 저장하는 중...")
# 2. 짧고 깔끔한 새 이름으로 다시 저장하면서, 내부 데이터 연결(문신)도 새롭게 고쳐줍니다.
onnx.save_model(
    model, 
    "skin_model.onnx", 
    save_as_external_data=True, 
    all_tensors_to_one_file=True, 
    location="skin_model.onnx.data" # ⭐️ 여기에 새로운 짝꿍 이름을 새겨줍니다!
)

print("✨ 완료! 이제 skin_model.onnx 와 skin_model.onnx.data 를 사용하실 수 있습니다.")
