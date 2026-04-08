import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from src.preprocess import load_hc3_data, clean_text
from src.perplexity_model import calculate_perplexity

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            "bert-base-uncased", num_labels=2
        )
        hidden_size = self.bert.config.hidden_size
        self.bert.classifier = torch.nn.Linear(hidden_size + 1, 2)

    def forward(self, input_ids, attention_mask, ppl, labels=None):
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.pooler_output

        ppl = ppl.unsqueeze(-1)
        fusion_emb = torch.cat([cls_emb, ppl], dim=-1)

        logits = self.bert.classifier(fusion_emb)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def get_ppl_features(texts, scaler=None):
    ppls = []
    for t in texts:
        try:
            ppl = calculate_perplexity(t)
            ppls.append(ppl)
        except:
            ppls.append(1000.0)  

    ppls = np.array(ppls).reshape(-1, 1)
    
    if scaler is None:
        scaler = StandardScaler()
        ppls = scaler.fit_transform(ppls)
    else:
        ppls = scaler.transform(ppls)
    return ppls.flatten(), scaler

def run_bert_ppl_fusion():
    import shutil
    shutil.rmtree("./fusion_ckpt", ignore_errors=True)
    print("\n===== 【强融合】BERT + PPL 端到端训练 =====")

    train, test = load_hc3_data()
    train_text = train["text"].apply(clean_text).tolist()
    test_text = test["text"].apply(clean_text).tolist()
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    train_ppl, scaler = get_ppl_features(train_text)
    test_ppl, _ = get_ppl_features(test_text, scaler=scaler)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertFusionModel().to(device)

    train_dataset = FusionDataset(train_text, train_ppl, y_train, tokenizer)
    test_dataset = FusionDataset(test_text, test_ppl, y_test, tokenizer)

    args = TrainingArguments(
        output_dir="./fusion_ckpt",
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
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_score = preds.predictions[:, 1]
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ 强融合模型 Acc: {acc:.4f}")
    return acc, y_test, y_pred, y_score

if __name__ == "__main__":
    run_bert_ppl_fusion()