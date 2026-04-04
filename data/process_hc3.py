import json
import pandas as pd
import re

def clean_text(text):
    """清洗文本：去除换行、多余空格、特殊符号"""
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_process_hc3(jsonl_path):
    """
    加载HC3 JSONL并处理成训练用DataFrame
    适配 all.jsonl 格式（每行一个JSON）
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item.get("question", "")
            
            # 处理人类回答
            human_answers = item.get("human_answers", [])
            if human_answers:
                human_text = clean_text(human_answers[0])
                if human_text:
                    rows.append({
                        "text": f"{question} {human_text}",
                        "label": 0
                    })
            
            # 处理AI回答
            chatgpt_answers = item.get("chatgpt_answers", [])
            if chatgpt_answers:
                ai_text = clean_text(chatgpt_answers[0])
                if ai_text:
                    rows.append({
                        "text": f"{question} {ai_text}",
                        "label": 1
                    })
    
    df = pd.DataFrame(rows)
    return df

def sample_balanced_data(df, n=10000):
    human_df = df[df["label"] == 0]
    ai_df = df[df["label"] == 1]
    human_sample = human_df.sample(n=min(n, len(human_df)), random_state=42)
    ai_sample = ai_df.sample(n=min(n, len(ai_df)), random_state=42)
    balanced_df = pd.concat([human_sample, ai_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

if __name__ == "__main__":
    # 这里已经自动适配你的 all.jsonl 文件
    json_file = "all.jsonl"
    output_csv = "HC3.csv"
    
    df = load_and_process_hc3(json_file)
    print(f"处理后总数据量：{len(df)}")
    print(f"人类数据：{len(df[df['label']==0])}，AI数据：{len(df[df['label']==1])}")
    
    balanced_df = sample_balanced_data(df, n=10000)
    print(f"均衡采样后数据量：{len(balanced_df)}")
    
    balanced_df.to_csv(output_csv, index=False, encoding="utf-8")
    print("✅ 处理完成！已生成 HC3.csv，可以训练模型了！")