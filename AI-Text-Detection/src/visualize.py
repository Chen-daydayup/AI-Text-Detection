import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

def get_tfidf_embedding(texts):
    vectorizer = TfidfVectorizer(max_features=10000)
    return vectorizer.fit_transform(texts).toarray()

def get_bert_embedding(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    embeddings = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            out = model(**inputs)
            embed = out.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embed)
    return np.array(embeddings)

def plot_tsne(embedding, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embedding)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[labels == 0, 0], emb_2d[labels == 0, 1], c="blue", label="Human", alpha=0.6)
    plt.scatter(emb_2d[labels == 1, 0], emb_2d[labels == 1, 1], c="red", label="AI", alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()