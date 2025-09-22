from mlflow.tracking import MlflowClient

client = MlflowClient()

# ตั้งชื่อ model ที่มีอยู่แล้ว
model_name = "breast-cancer-classifier-prod"

# กำหนด version ที่ต้องการ
version = 2   # เช่นถ้า model version 1

# สร้าง alias staging ให้ model นี้
client.set_registered_model_alias(
    name=model_name,
    alias="staging",
    version=version
)

print(f"Alias 'staging' set for model '{model_name}' version {version}")
