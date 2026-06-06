import os
import pickle
import shutil

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from src.preprocess import load_hc3_data, clean_text
from src.perplexity_model import calculate_perplexity


device = "cuda" if torch.cuda.is_available() else "cpu"

FUSION_CKPT_DIR = "./fusion_ckpt"
FUSION_MODEL_PATH = os.path.join(FUSION_CKPT_DIR, "fusion_model.pt")
PPL_SCALER_PATH = os.path.join(FUSION_CKPT_DIR, "ppl_scaler.pkl")
BERT_MODEL_NAME = "bert-base-uncased"


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, ppls, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        self.ppls = torch.tensor(ppls, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "ppl": self.ppls[idx],
            "labels": self.labels[idx]
        }
        return item


class BertFusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=2
        )

        hidden_size = self.bert.config.hidden_size

        # 原始 BERT 分类头输入维度是 hidden_size
        # 现在拼接 1 维 PPL 特征，所以输入维度变成 hidden_size + 1
        self.bert.classifier = torch.nn.Linear(hidden_size + 1, 2)

    def forward(self, input_ids, attention_mask, ppl, labels=None):
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # pooler_output: [batch_size, hidden_size]
        cls_emb = outputs.pooler_output

        # ppl: [batch_size] -> [batch_size, 1]
        ppl = ppl.unsqueeze(-1)

        # fusion_emb: [batch_size, hidden_size + 1]
        fusion_emb = torch.cat([cls_emb, ppl], dim=-1)

        logits = self.bert.classifier(fusion_emb)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }


def get_ppl_features(texts, scaler=None):
    """
    计算文本 PPL 特征，并进行标准化。

    如果 scaler=None：
        说明当前是训练阶段，对训练集 fit_transform。

    如果 scaler 不为 None：
        说明当前是测试/可视化阶段，只 transform，不能重新 fit。
    """
    ppls = []

    for t in texts:
        try:
            ppl = calculate_perplexity(t)
            ppls.append(ppl)
        except Exception:
            # 防止个别文本计算 PPL 失败导致整个实验中断
            ppls.append(1000.0)

    ppls = np.array(ppls).reshape(-1, 1)

    if scaler is None:
        scaler = StandardScaler()
        ppls = scaler.fit_transform(ppls)
    else:
        ppls = scaler.transform(ppls)

    return ppls.flatten(), scaler


def run_bert_ppl_fusion():
    """
    训练 BERT + PPL 端到端融合模型。

    训练完成后会保存：
    1. ./fusion_ckpt/fusion_model.pt
       训练后的 Fusion 模型参数。

    2. ./fusion_ckpt/ppl_scaler.pkl
       训练集上 fit 出来的 PPL 标准化器。
    """

    # 每次重新训练前，删除旧的 fusion_ckpt
    shutil.rmtree(FUSION_CKPT_DIR, ignore_errors=True)
    os.makedirs(FUSION_CKPT_DIR, exist_ok=True)

    print("\n===== 【强融合】BERT + PPL 端到端训练 =====")

    train, test = load_hc3_data()

    train_text = train["text"].apply(clean_text).tolist()
    test_text = test["text"].apply(clean_text).tolist()

    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    print("正在计算训练集 PPL 特征...")
    train_ppl, scaler = get_ppl_features(train_text)

    print("正在计算测试集 PPL 特征...")
    test_ppl, _ = get_ppl_features(test_text, scaler=scaler)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    model = BertFusionModel().to(device)

    train_dataset = FusionDataset(
        texts=train_text,
        ppls=train_ppl,
        labels=y_train,
        tokenizer=tokenizer
    )

    test_dataset = FusionDataset(
        texts=test_text,
        ppls=test_ppl,
        labels=y_test,
        tokenizer=tokenizer
    )

    args = TrainingArguments(
        output_dir=FUSION_CKPT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        learning_rate=1e-5,
        eval_strategy="no",
        save_strategy="no",
        disable_tqdm=False,
        report_to="none",
        seed=42,
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset
    )

    trainer.train()

    torch.save(
        model.state_dict(),
        FUSION_MODEL_PATH
    )

    with open(PPL_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ 已保存训练后的 Fusion 模型到：{FUSION_MODEL_PATH}")
    print(f"✅ 已保存 PPL 标准化器到：{PPL_SCALER_PATH}")

    preds = trainer.predict(test_dataset)

    y_pred = np.argmax(preds.predictions, axis=1)

    y_score = preds.predictions[:, 1]

    acc = accuracy_score(y_test, y_pred)

    print(f"✅ 强融合模型 Acc: {acc:.4f}")

    return acc, y_test, y_pred, y_score


if __name__ == "__main__":
    run_bert_ppl_fusion()
