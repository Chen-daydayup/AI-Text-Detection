import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import load_hc3_data, clean_text

def get_length_features(df):
    lengths = []
    for text in df["text"]:
        cleaned = clean_text(str(text))
        lengths.append(len(cleaned))
    return np.array(lengths).reshape(-1, 1)

def run_length_lr():
    print("\n===== 正在运行 Length-only Baseline =====")
    train, test = load_hc3_data()

    # 直接调用外部函数
    X_train = get_length_features(train)
    X_test = get_length_features(test)
    y_train, y_test = train["label"], test["label"]

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    print(f"Length-only LR 准确率: {acc:.4f}")
    return acc, y_test, y_pred, y_score