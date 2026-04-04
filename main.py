import os
from src.evaluate import compare_models

# 自动创建文件夹
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

if __name__ == "__main__":
    compare_models()