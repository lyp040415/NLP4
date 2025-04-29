import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tqdm import tqdm

# 配置参数
corpus_dir = "D:/课程/大四/大四下/自然语言处理/第四次作业/jyxstxtqj_downcc.com"
model_path = "style_finetuned.keras"
# corpus_dir = "F:/人民日报数据集/renmin1949-1978-master"
# model_path = "pretrain.keras"
seq_length = 100  # 必须与训练时相同

# 语料加载函数（与训练代码一致）
def load_corpus(folder_path):
    text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="gb18030", errors="ignore") as f:
                    text += f.read()
    return text.replace("\r\n", "\n")

# 重建字符映射
print("Rebuilding vocabulary...")
corpus_text = load_corpus(corpus_dir)
vocab = sorted(set(corpus_text))

# 创建相同的StringLookup层
char2idx = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
idx2char = tf.keras.layers.StringLookup(
    vocabulary=char2idx.get_vocabulary(), invert=True, mask_token=None)

# 改进版文本生成函数
def generate_text(model, start_string, num_generate=500, temperature=1.0):
    # 转换输入文本为ID序列
    input_chars = tf.strings.unicode_split([start_string], 'UTF-8').numpy()[0]
    input_ids = char2idx(input_chars)[tf.newaxis, :]
    
    generated = []
    for _ in tqdm(range(num_generate), desc="Generating"):
        # 确保输入不超过训练时的序列长度
        truncated_input = input_ids[:, -seq_length:]
        
        # 获取模型预测
        predictions = model(truncated_input)
        
        # 提取最后一个时间步的logits
        last_step_logits = predictions[:, -1, :]
        scaled_logits = last_step_logits / temperature
        
        # 采样下一个字符
        predicted_id = tf.random.categorical(scaled_logits, num_samples=1)
        decoded_char = idx2char(predicted_id).numpy()[0][0].decode('utf-8')
        
        # 更新输入序列
        input_ids = tf.concat([input_ids, predicted_id], axis=-1)
        generated.append(decoded_char)

    return start_string + ''.join(generated)

# 加载模型
print("\nLoading fine-tuned model...")
model = tf.keras.models.load_model(model_path)

# 生成示例
generated_text = generate_text(
    model,
    start_string="郭靖对黄蓉说",
    num_generate=1000,
    temperature= 0.7  # 推荐值：0.5-1.2
)

print("\n生成结果:\n" + "="*50)
print(generated_text)
print("="*50)

output_path = os.path.join(os.path.dirname(model_path), "LSTM_text.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(generated_text)
print(f"\n文本已保存至: {output_path}")