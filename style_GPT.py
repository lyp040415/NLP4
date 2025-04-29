import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import torch

output_dir = r"D:/GPT"
os.makedirs(output_dir, exist_ok=True)

if __name__ == '__main__':
    # 1. 加载金庸小说txt文件
    def load_texts_from_folder(folder_path):
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                except UnicodeDecodeError:
                    with open(filepath, 'r', encoding='gbk', errors='ignore') as file:
                        texts.append(file.read())
        return texts

    # 2. 文本预处理（去除无关字符、只保留正文）
    def preprocess_text(text):
        text = re.sub(r'\s+', '\n', text)  # 多个空格、Tab、换行统一成单个换行
        text = re.sub(r'[^\u4e00-\u9fa5。\n！？]', '', text)  # 只保留中文字符和标点
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]  # 去掉太短的行
        return paragraphs

    # 3. 把所有小说整合成训练数据列表
    folder_path = r"D:\课程\大四\大四下\自然语言处理\第四次作业\jyxstxtqj_downcc.com"
    all_texts = load_texts_from_folder(folder_path)

    all_paragraphs = []
    for text in all_texts:
        paragraphs = preprocess_text(text)
        all_paragraphs.extend(paragraphs)

    print(f"总共有 {len(all_paragraphs)} 个段落可用于训练")

    # 4. 准备 Huggingface Dataset
    dataset = Dataset.from_dict({"text": all_paragraphs})

    # 5. 加载预训练模型和分词器
    pretrained_model_name = "uer/gpt2-chinese-cluecorpussmall"  # 中文GPT2小型模型示例
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

    # 6. Tokenize数据
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. 设置DataCollator (自回归建模，不使用MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    # 8. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="tensorboard",
        dataloader_num_workers=4,  # 这里设置数据加载器的线程数
        # 这里不需要device参数，Trainer会自动使用可用的GPU/CPU
    )

    # 9. 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 10. 开始训练
    trainer.train()

    # 11. 保存最终模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

