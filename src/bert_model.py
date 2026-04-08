import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from src.preprocess import load_hc3_data, clean_text

class AIDetectDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def run_bert_finetune():
    import shutil
    shutil.rmtree("./bert_finetuned", ignore_errors=True)
    print("\n===== 正在运行 BERT 微调 =====")
    train, test = load_hc3_data()
    
    train_text = train['text'].apply(clean_text).tolist()
    test_text = test['text'].apply(clean_text).tolist()
    y_train = train['label'].tolist()
    y_test = test['label'].tolist()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    train_dataset = AIDetectDataset(train_text, y_train, tokenizer)
    test_dataset = AIDetectDataset(test_text, y_test, tokenizer)
    
    args = TrainingArguments(
        output_dir="./bert_ckpt",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        learning_rate=1e-5,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no", 
        disable_tqdm=False,
        report_to="none",
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    model.save_pretrained("./bert_finetuned")  
    tokenizer.save_pretrained("./bert_finetuned")  
    
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_score = preds.predictions[:, 1]
    acc = accuracy_score(y_test, y_pred)
    
    print(f"BERT Fine-tune 准确率: {acc:.4f}")
    print("✅ 微调后的模型已保存到：./bert_finetuned")
    
    return acc, y_test, y_pred, y_score