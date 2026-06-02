from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'skin_model.onnx'
model_rp5 = 'skin_model_rp5.onnx'

print("모델 압축을 시작합니다. 잠시만 기다려주세요.")

# 동적 양 자화시키기기 
quantize_dynamic(model_fp32, model_rp5, weight_type=QuantType.QUInt8)

print(f"압축 완료. 파일이 생성되었습니다: {model_rp5}")