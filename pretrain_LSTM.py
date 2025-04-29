# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import glob
# import tensorflow as tf
# from tqdm import tqdm

# # —— 一、屏蔽日志 & 关闭 XLA JIT —— 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')
# tf.config.optimizer.set_jit(False)

# # —— 二、超参数 —— 
# DATA_DIR      = r"F:\人民日报数据集\renmin1949-1978-master"
# MAX_CHARS     = 10_000_000    # 只用前 500 万字符做测试
# SEQ_LENGTH    = 100           # 序列长度
# BATCH_SIZE    = 16            # 批大小
# EPOCHS        = 5            # 训练轮数
# EMBEDDING_DIM = 64           # 词嵌入维度
# RNN_UNITS     = 64           # LSTM 单元数

# # —— 三、加载 & 裁剪语料 —— 
# print("Loading corpus...")
# txt_files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)

# chars, char_count = [], 0
# for fn in txt_files:
#     with open(fn, 'r', encoding='utf-8') as f:
#         for line in f:
#             chars.append(line)
#             char_count += len(line)
#             if char_count >= MAX_CHARS:
#                 break
#     if char_count >= MAX_CHARS:
#         break

# corpus = "".join(chars)
# print(f"Loaded {len(corpus):,} characters (trimmed)")

# # —— 四、构建词表 & 静态哈希表映射 —— 
# vocab = sorted(set(corpus))
# vocab_size = len(vocab)
# print(f"Vocabulary size: {vocab_size}")

# # 用 StaticHashTable 建立 char->idx 映射，确保索引从 0 开始，到 vocab_size-1
# keys = tf.constant(vocab)
# vals = tf.constant(list(range(vocab_size)), dtype=tf.int64)
# initializer = tf.lookup.KeyValueTensorInitializer(keys, vals)
# table = tf.lookup.StaticHashTable(initializer, default_value=0)

# # —— 五、构造 Dataset —— 
# # 1) 切分字符
# chars_ds = tf.data.Dataset.from_tensor_slices(
#     tf.strings.unicode_split(corpus, 'UTF-8')
# )
# # 2) 映射为 ID
# ids_ds = chars_ds.map(lambda ch: table.lookup(ch),
#                       num_parallel_calls=tf.data.AUTOTUNE)

# # 3) 按 SEQ_LENGTH+1 分批，拆成 (input, target)
# seq_ds = ids_ds.batch(SEQ_LENGTH + 1, drop_remainder=True)
# def split_input_target(seq):
#     return seq[:-1], seq[1:]
# dataset = seq_ds.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)

# # 4) 打乱、批量、预取
# train_ds = (
#     dataset
#     .shuffle(5_000)
#     .batch(BATCH_SIZE, drop_remainder=True)
#     .prefetch(tf.data.AUTOTUNE)
# )
# steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
# print(f"Dataset ready: {steps_per_epoch:,} batches per epoch")

# # —— 六、模型搭建 —— 
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
#     tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True),
#     tf.keras.layers.Dense(vocab_size)
# ])
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# )

# # —— 七、tqdm 进度回调 —— 
# class TqdmCallback(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.prog = tqdm(total=steps_per_epoch,
#                          desc=f"Epoch {epoch+1}/{EPOCHS}",
#                          unit="batch")
#     def on_train_batch_end(self, batch, logs=None):
#         self.prog.set_postfix(loss=f"{logs['loss']:.4f}")
#         self.prog.update(1)
#     def on_epoch_end(self, epoch, logs=None):
#         self.prog.close()

# # —— 八、训练 & 保存 —— 
# print("Starting training...")
# model.fit(
#     train_ds,
#     epochs=EPOCHS,
#     callbacks=[TqdmCallback()],
#     verbose=0
# )

# model.save("pretrain.keras")
# print("Model saved as pretrain.keras")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import tensorflow as tf
from tqdm import tqdm

# —— 一、屏蔽日志 & 关闭 XLA JIT ——
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(False)

# —— 二、超参数 ——
DATA_DIR       = r"F:\人民日报数据集\renmin1949-1978-master"
MAX_CHARS      = 10_000_000     # 使用前1000万字符
SEQ_LENGTH     = 100            # 序列长度
BATCH_SIZE     = 32             # 批大小
EPOCHS         = 5             # 训练轮数
EMBEDDING_DIM  = 128            # 词嵌入维度
RNN_UNITS      = 256            # LSTM单元数
NUM_RNN_LAYERS = 2              # LSTM层数
DENSE_UNITS    = 256            # 全连接层维度
DROPOUT_RATE   = 0.1            # Dropout比例
LEARNING_RATE  = 0.001          # 学习率

# —— 三、加载 & 裁剪语料 ——
print("Loading corpus...")
txt_files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)

chars, char_count = [], 0
for fn in txt_files:
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            chars.append(line)
            char_count += len(line)
            if char_count >= MAX_CHARS:
                break
    if char_count >= MAX_CHARS:
        break

corpus = "".join(chars)[:MAX_CHARS]  # 确保精确截断
print(f"Loaded {len(corpus):,} characters (trimmed)")

# —— 四、构建词表 & 静态哈希表映射 ——
vocab = sorted(set(corpus))
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# 建立字符到索引的映射
keys = tf.constant(vocab)
vals = tf.constant(list(range(vocab_size)), dtype=tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals), 
    default_value=0
)

# —— 五、构造 Dataset ——
# 1) 切分字符
chars_ds = tf.data.Dataset.from_tensor_slices(
    tf.strings.unicode_split(corpus, 'UTF-8')
)
# 2) 映射为 ID
ids_ds = chars_ds.map(lambda ch: table.lookup(ch),
                     num_parallel_calls=tf.data.AUTOTUNE)

# 3) 按 SEQ_LENGTH+1 分批
seq_ds = ids_ds.batch(SEQ_LENGTH + 1, drop_remainder=True)

def split_input_target(seq):
    return seq[:-1], seq[1:]

dataset = seq_ds.map(split_input_target, 
                    num_parallel_calls=tf.data.AUTOTUNE)

# 4) 打乱、批量、预取
train_ds = (
    dataset
    .shuffle(10_000)  # 增大shuffle buffer
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Dataset ready: {steps_per_epoch:,} batches per epoch")

# —— 六、复杂模型搭建 ——
def build_model():
    model = tf.keras.Sequential([
        # 增强的词嵌入层（指定input_length以触发权重构建）
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
            input_length=SEQ_LENGTH  # 关键修复：明确指定序列长度
        ),
        
        # 堆叠双向LSTM（双向有助于理解上下文）
        *[tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                RNN_UNITS,
                return_sequences=True,
                dropout=DROPOUT_RATE,
                recurrent_dropout=DROPOUT_RATE
            )
        ) for _ in range(NUM_RNN_LAYERS)],
        
        # 中间全连接层
        tf.keras.layers.Dense(DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        # 输出层
        tf.keras.layers.Dense(vocab_size)
    ])
    
    # 带梯度裁剪的优化器
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipvalue=5.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

model = build_model()
model.build(input_shape=(BATCH_SIZE, SEQ_LENGTH))  # 关键修复：手动触发权重初始化
model.summary()  # 现在将显示完整参数信息

# —— 七、训练进度回调 ——
class TqdmCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.prog = tqdm(total=steps_per_epoch,
                         desc=f"Epoch {epoch+1}/{EPOCHS}",
                         unit="batch")
    def on_train_batch_end(self, batch, logs=None):
        self.prog.set_postfix(
            loss=f"{logs['loss']:.4f}",
            acc=f"{logs['accuracy']:.2%}"
        )
        self.prog.update(1)
    def on_epoch_end(self, epoch, logs=None):
        self.prog.close()

# —— 八、训练 & 保存 ——
print("Starting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[TqdmCallback()],
    verbose=0
)

model.save("enhanced_pretrain.keras")
print("Model saved as enhanced_pretrain.keras")