import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

# 设置参数
corpus_dir = "D:/课程/大四/大四下/自然语言处理/第四次作业/jyxstxtqj_downcc.com"
pretrained_model_path = "pretrain.keras"
fine_tuned_model_path = "style_finetuned.keras"
seq_length = 100
batch_size = 64
buffer_size = 10000
epochs = 10

# 加载语料
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

print("Loading corpus...")
text = load_corpus(corpus_dir)
print(f"Corpus length: {len(text)} characters")

# 构建字符集
vocab = sorted(set(text))
print(f"Vocabulary size: {len(vocab)}")

# 建立映射
char2idx = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
idx2char = tf.keras.layers.StringLookup(vocabulary=char2idx.get_vocabulary(), invert=True, mask_token=None)

# 向量化文本
def text_to_ids(text):
    return char2idx(tf.strings.unicode_split(text, input_encoding="UTF-8"))

text_as_int = text_to_ids(text)

# 构建训练数据
def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# 加载预训练模型
print("Loading pre-trained model...")
model = load_model(pretrained_model_path)
print("Model loaded.")

# 编译模型
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# 自定义训练循环（带进度条）
print("Starting fine-tuning...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    progbar = tqdm(dataset, desc="Training", unit="batch")
    for (batch_n, (inp, target)) in enumerate(progbar):
        with tf.GradientTape() as tape:
            predictions = model(inp)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        progbar.set_postfix(loss=loss.numpy())

# 保存微调后的模型
model.save(fine_tuned_model_path)
print(f"\nFine-tuned model saved to {fine_tuned_model_path}")