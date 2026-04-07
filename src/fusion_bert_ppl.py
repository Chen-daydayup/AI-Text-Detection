import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from src.preprocess import load_hc3_data, clean_text
from src.perplexity_model import calculate_perplexity

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_bert_sentence_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    embeddings = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length"
            ).to(device)
            out = model(**inputs)
            cls_emb = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_emb)
    
    return np.array(embeddings)

def get_fusion_features(texts):
    bert_emb = get_bert_sentence_embeddings(texts)
    
    ppls = []
    for t in texts:
        ppl = calculate_perplexity(t)
        ppls.append(ppl)
    ppl_arr = np.array(ppls).reshape(-1, 1)
    
    scaler = StandardScaler()
    ppl_scaled = scaler.fit_transform(ppl_arr)
    
    fusion = np.concatenate([bert_emb, ppl_scaled], axis=1)
    
    return fusion

def run_bert_ppl_fusion():
    print("\n===== 正在运行 BERT + Perplexity 融合模型 =====")
    
    train, test = load_hc3_data()
    
    train_text = train["text"].apply(clean_text).tolist()
    test_text = test["text"].apply(clean_text).tolist()
    
    y_train = train["label"].values
    y_test = test["label"].values

    print("正在生成训练集融合特征...")
    X_train = get_fusion_features(train_text)
    
    print("正在生成测试集融合特征...")
    X_test = get_fusion_features(test_text)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    print(f"BERT + PPL 融合模型 准确率: {acc:.4f}")
    return acc, y_test, y_pred, y_score