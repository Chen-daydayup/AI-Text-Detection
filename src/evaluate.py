from src.tfidf_model import run_tfidf_lr
from src.perplexity_model import run_perplexity_lr
from src.bert_model import run_bert_finetune
from src.length_model import run_length_lr
from src.fusion_bert_ppl import run_bert_ppl_fusion, get_fusion_features
from src.preprocess import load_hc3_data, clean_text

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
import torch
import os
import pandas as pd

os.makedirs("results", exist_ok=True)

def plot_metrics(y_true, y_pred, y_score, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix | {model_name}")
    plt.savefig(f"results/{model_name}_cm.png", bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve | {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"results/{model_name}_roc.png", bbox_inches="tight")
    plt.close()
    return roc_auc

def analyze_text_length():
    print("\n" + "="*60)
    print("📏 文本长度统计分析")
    print("="*60)
    
    train, test = load_hc3_data()
    df = pd.concat([train, test], ignore_index=True)
    
    human_lengths = []
    ai_lengths = []
    
    for text, label in zip(df['text'], df['label']):
        cleaned = clean_text(str(text))
        length = len(cleaned)
        if label == 0:
            human_lengths.append(length)
        else:
            ai_lengths.append(length)
    
    avg_human = np.mean(human_lengths)
    avg_ai = np.mean(ai_lengths)
    
    print(f"{'类型':<10} {'平均长度':<15}")
    print("-" * 25)
    print(f"{'Human':<10} {avg_human:.1f} 字符")
    print(f'{"AI":<10} {avg_ai:.1f} 字符')
    print("="*60)

    plt.figure(figsize=(8, 5))
    bins = np.arange(0, 1001, 20)
    
    sns.histplot(
        human_lengths, label="Human", color="#4285F4", alpha=0.6, bins=bins
    )
    sns.histplot(
        ai_lengths, label="AI", color="#EA4335", alpha=0.6, bins=bins
    )

    plt.axvline(avg_human, color='#4285F4', linestyle='--', linewidth=2)
    plt.axvline(avg_ai, color='#EA4335', linestyle='--', linewidth=2)

    plt.text(avg_human + 30, plt.gca().get_ylim()[1]*0.8,
             f'Human mean: {avg_human:.1f}',
             color='#4285F4', fontweight='bold')
    plt.text(avg_ai + 30, plt.gca().get_ylim()[1]*0.7,
             f'AI mean: {avg_ai:.1f}',
             color='#EA4335', fontweight='bold')

    plt.xlabel("Text Length (characters)")
    plt.ylabel("Count")
    plt.title("Text Length Distribution: Human vs AI")
    plt.legend()
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig("results/length_distribution.png", dpi=150)
    plt.close()
    print("✅ 长度分布图已保存\n")

def visualize_all_tsne():
    print("\n" + "="*60)
    print("🧠 生成 t-SNE 可视化图表...")
    print("="*60)

    train, test = load_hc3_data()
    df = pd.concat([train, test], ignore_index=True)
    
    df_human = df[df['label'] == 0].sample(n=3000, random_state=42)  
    df_ai = df[df['label'] == 1].sample(n=3000, random_state=42)     
    df_sampled = pd.concat([df_human, df_ai], ignore_index=True)    
    
    texts = [clean_text(str(t)) for t in df_sampled["text"]]
    labels = df_sampled["label"].values
    
    device = "cpu"  

    print("📊 生成 TF-IDF t-SNE...")
    tfidf = TfidfVectorizer(max_features=10000, max_df=0.9, ngram_range=(1,2))
    emb_tfidf = tfidf.fit_transform(texts).toarray()
    tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=30)
    emb2d = tsne.fit_transform(emb_tfidf)
    plt.figure(figsize=(8,6))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c="blue", label="Human", alpha=0.4, s=12)
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c="red", label="AI", alpha=0.4, s=12)
    plt.title("t-SNE (TF-IDF Features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/tsne_tfidf.png", dpi=150)
    plt.close()

    print("📊 生成 BERT t-SNE...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    emb_bert = []
    with torch.no_grad():
        for txt in texts:
            toks = tokenizer(txt, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            toks = {k:v.to(device) for k,v in toks.items()}
            out = model(**toks)
            emb = out.last_hidden_state.mean(1).squeeze().cpu().numpy()
            emb_bert.append(emb)
    emb_bert = np.array(emb_bert)
    emb2d = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(emb_bert)
    plt.figure(figsize=(8,6))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c="blue", label="Human", alpha=0.4, s=12)
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c="red", label="AI", alpha=0.4, s=12)
    plt.title("t-SNE (BERT Embeddings)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/tsne_bert.png", dpi=150)
    plt.close()

    print("📊 生成 BERT+PPL Fusion t-SNE...")
    emb_fusion = get_fusion_features(texts)
    emb2d = TSNE(n_components=2, random_state=42, init="random", perplexity=30).fit_transform(emb_fusion)
    plt.figure(figsize=(8,6))
    plt.scatter(emb2d[labels==0,0], emb2d[labels==0,1], c="blue", label="Human", alpha=0.4, s=12)
    plt.scatter(emb2d[labels==1,0], emb2d[labels==1,1], c="red", label="AI", alpha=0.4, s=12)
    plt.title("t-SNE (BERT + Perplexity Fusion)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/tsne_bert_ppl_fusion.png", dpi=150)
    plt.close()

    print("✅ t-SNE 图表全部生成完成！\n")

def get_top_tfidf_features():
    print("\n" + "="*60)
    print("🔍 Top 20 TF-IDF 特征词 (AI vs Human 中文专用)")
    print("="*60)

    train, test = load_hc3_data()
    df = pd.concat([train, test])
    texts = [clean_text(str(t)) for t in df["text"]]
    labels = df["label"].values

    tfidf = TfidfVectorizer(max_features=10000, max_df=0.9,ngram_range=(1,2))
    X = tfidf.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    feats = tfidf.get_feature_names_out()
    coef = model.coef_[0]

    top_ai = [feats[i] for i in coef.argsort()[-20:]][::-1]
    top_human = [feats[i] for i in coef.argsort()[:20]]

    print("\n===== Top 20 AI 标志性词汇 =====")
    for i, w in enumerate(top_ai, 1):
        print(f"{i:2d}. {w}")
    
    print("\n===== Top 20 Human 标志性词汇 =====")
    for i, w in enumerate(top_human, 1):
        print(f"{i:2d}. {w}")

    with open("results/top_tfidf_features.txt", "w", encoding="utf-8") as f:
        f.write("Top 20 AI 标志性词汇：\n")
        f.write("\n".join(top_ai))
        f.write("\n\nTop 20 Human 标志性词汇：\n")
        f.write("\n".join(top_human))
    
    print("\n✅ 特征词已保存到 results/top_tfidf_features.txt\n")

def save_error_examples(model_name, y_true, y_pred, texts):
    fp_texts = []
    fn_texts = []

    for t, p, txt in zip(y_true, y_pred, texts):
        if len(fp_texts) >= 10 and len(fn_texts) >= 10:
            break
        txt = txt[:200] + "..." if len(txt) > 200 else txt
        if t == 0 and p == 1 and len(fp_texts) < 10:
            fp_texts.append(txt)
        if t == 1 and p == 0 and len(fn_texts) < 10:
            fn_texts.append(txt)

    with open(f"results/error_samples_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} 错误样本分析 ===\n\n")
        f.write(f"False Positive (真Human → 预测AI) 共10条：\n\n")
        for i, t in enumerate(fp_texts, 1):
            f.write(f"{i:2d}. {t}\n\n")

        f.write(f"\nFalse Negative (真AI → 预测Human) 共10条：\n\n")
        for i, t in enumerate(fn_texts, 1):
            f.write(f"{i:2d}. {t}\n\n")

    print(f"✅ {model_name} FP/FN 样本已保存！")

def compare_models():
    print("=" * 70)
    print("🚀 AI 生成文本检测：五模型对比实验")
    print("=" * 70)

    #  数据分析与可视化
    analyze_text_length()
    visualize_all_tsne()
    get_top_tfidf_features()

    train, test = load_hc3_data()
    test_texts = [clean_text(str(t)) for t in test["text"]]
    y_test = test["label"].values
    
    # 运行各模型并保存错误样本
    acc4, y_true4, y_pred4, y_score4 = run_length_lr()
    save_error_examples("Length", y_true4, y_pred4, test_texts)

    acc2, y_true2, y_pred2, y_score2 = run_perplexity_lr()
    save_error_examples("Perplexity", y_true2, y_pred2, test_texts)

    acc1, y_true1, y_pred1, y_score1 = run_tfidf_lr()
    save_error_examples("TF-IDF-LR", y_true1, y_pred1, test_texts)

    acc3, y_true3, y_pred3, y_score3 = run_bert_finetune()
    save_error_examples("BERT", y_true3, y_pred3, test_texts)

    acc5, y_true5, y_pred5, y_score5 = run_bert_ppl_fusion()
    save_error_examples("BERT-PPL-Fusion", y_true5, y_pred5, test_texts)

    # 绘制性能对比图表
    auc4 = plot_metrics(y_true4, y_pred4, y_score4, "Length-only-LR")
    auc2 = plot_metrics(y_true2, y_pred2, y_score2, "Perplexity-LR")
    auc1 = plot_metrics(y_true1, y_pred1, y_score1, "TF-IDF-LR")
    auc3 = plot_metrics(y_true3, y_pred3, y_score3, "BERT")
    auc5 = plot_metrics(y_true5, y_pred5, y_score5, "BERT-PPL-Fusion")

    # 最终结果对比表
    print("\n" + "=" * 70)
    print("📊 最终模型性能对比表")
    print("=" * 70)
    print(f"{'模型':<22} {'准确率':<12} {'AUC':<12}")
    print("-" * 70)
    print(f"{'Length-only LR':<22} {acc4:<12.4f} {auc4:<12.4f}")
    print(f"{'Perplexity + LR':<22} {acc2:<12.4f} {auc2:<12.4f}")
    print(f"{'TF-IDF + LR':<22} {acc1:<12.4f} {auc1:<12.4f}")
    print(f"{'BERT Fine-tune':<22} {acc3:<12.4f} {auc3:<12.4f}")
    print(f"{'BERT+PPL Fusion':<22} {acc5:<12.4f} {auc5:<12.4f}")
    print("=" * 70)

    with open("results/experiment_result.txt", "w", encoding="utf-8") as f:
        f.write("AI 文本检测模型对比结果\n")
        f.write("="*40 + "\n")
        f.write(f"Length-only LR:   Acc={acc4:.4f}, AUC={auc4:.4f}\n")
        f.write(f"Perplexity + LR:  Acc={acc2:.4f}, AUC={auc2:.4f}\n")
        f.write(f"TF-IDF + LR:      Acc={acc1:.4f}, AUC={auc1:.4f}\n")
        f.write(f"BERT Fine-tune:   Acc={acc3:.4f}, AUC={auc3:.4f}\n")
        f.write(f"BERT+PPL Fusion:  Acc={acc5:.4f}, AUC={auc5:.4f}\n")

    print("\n✅ 全部任务完成！")

if __name__ == "__main__":
    compare_models()