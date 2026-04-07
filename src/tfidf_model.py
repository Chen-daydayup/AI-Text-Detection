from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import load_hc3_data, clean_text

def run_tfidf_lr():
    print("\n===== 正在运行 TF-IDF + LR =====")
    train, test = load_hc3_data()
    
    # 文本清洗
    train_text = train['text'].apply(clean_text).tolist()
    test_text = test['text'].apply(clean_text).tolist()
    y_train, y_test = train['label'], test['label']
    
    # TF-IDF 特征提取
    tfidf = TfidfVectorizer(max_features=10000, stop_words=None, max_df=0.9,ngram_range=(1,2))
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]  # 用于绘制ROC曲线
    acc = accuracy_score(y_test, y_pred)
    
    print(f"TF-IDF + LR 准确率: {acc:.4f}")
    
    return acc, y_test, y_pred, y_score