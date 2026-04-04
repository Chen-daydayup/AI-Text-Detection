import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.preprocess import load_hc3_data, clean_text

# 全局加载模型（避免重复加载）
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer.pad_token = tokenizer.eos_token

def calculate_perplexity(text):
    """计算文本困惑度：AI文本通常更低"""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        return min(ppl, 10000)  # 防止异常值
    except:
        return 5000.0

def run_perplexity_lr():
    print("\n===== 正在运行 Perplexity + Logistic Regression =====")
    train, test = load_hc3_data()
    
    # 提取困惑度特征
    print("正在计算训练集困惑度...")
    train["ppl"] = train["text"].apply(lambda x: calculate_perplexity(clean_text(x)))
    print("正在计算测试集困惑度...")
    test["ppl"] = test["text"].apply(lambda x: calculate_perplexity(clean_text(x)))
    
    # 构建特征
    X_train = train["ppl"].values.reshape(-1, 1)
    X_test = test["ppl"].values.reshape(-1, 1)
    y_train, y_test = train["label"], test["label"]
    
    # 逻辑回归
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    y_score = lr.predict_proba(X_test)[:, 1]  # 输出AI类别概率
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Perplexity + LR 准确率: {acc:.4f}")
    
    return acc, y_test, y_pred, y_score