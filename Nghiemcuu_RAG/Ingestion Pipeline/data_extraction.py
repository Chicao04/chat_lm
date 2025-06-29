import os
import yaml
import pickle
from multimodal_data import load_multimodal_data

# Đảm bảo thư mục scripts/ tồn tại
os.makedirs("scripts", exist_ok=True)

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load dữ liệu từ .txt, .pdf
documents = load_multimodal_data(config['data_raw_path'])

# Lưu lại dữ liệu đã trích xuất
with open("scripts/extracted.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Data extraction completed, saved to scripts/extracted.pkl")
