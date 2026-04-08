import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

def load_hc3_data(data_path="data/HC3.csv"):
    """
    加载HC3数据集
    human: 0, ai: 1
    均衡采样：各10000条
    """
    df = pd.read_csv(data_path)
    
    # 确保标签列正确（兼容原版HC3）
    if 'label' not in df.columns:
        df['label'] = df['source'].apply(lambda x: 1 if x == 'chatgpt' else 0)
    
    # 均衡采样 10000+10000
    human_data = df[df['label'] == 0].sample(n=10000, random_state=42, replace=False)
    ai_data = df[df['label'] == 1].sample(n=10000, random_state=42, replace=False)
    data = pd.concat([human_data, ai_data]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 8:2 划分训练集/测试集
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    return train_df, test_df

def clean_text(text):
    """基础文本清洗"""
    text = str(text).lower().strip()
    return text