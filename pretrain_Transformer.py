import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm

# —— 一、屏蔽日志 & 关闭 XLA JIT —— 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(False)

# —— 二、超参数 —— 
DATA_DIR      = r"F:\人民日报数据集\renmin1949-1978-master"
MAX_CHARS     = 5_000_000    # 只用前 1000 万字符做测试
SEQ_LENGTH    = 100           # 序列长度（包括下一字预测）
BATCH_SIZE    = 32            # 批大小
EPOCHS        = 1             # 训练轮数
VOCAB_SIZE    = 20000         # 词表大小
EMBEDDING_DIM = 64            # 词嵌入维度
NUM_HEADS     = 8             # 多头注意力头数
FF_DIM        = 256           # 前馈网络维度
NUM_LAYERS    = 6             # Transformer 层数

# —— 三、加载 & 裁剪语料 —— 
print("Loading corpus...")
txt_files = glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)
chars, count = [], 0
for fn in txt_files:
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            chars.append(line)
            count += len(line)
            if count >= MAX_CHARS:
                break
    if count >= MAX_CHARS:
        break
corpus = "".join(chars)
print(f"Loaded {len(corpus):,} characters (trimmed)")

# —— 四、构建词表 & 静态哈希表映射 —— 
vocab = sorted(set(corpus))
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
keys = tf.constant(vocab)
vals = tf.constant(list(range(vocab_size)), dtype=tf.int64)
initializer = tf.lookup.KeyValueTensorInitializer(keys, vals)
table = tf.lookup.StaticHashTable(initializer, default_value=0)

# —— 五、构造 Dataset —— 
chars_ds = tf.data.Dataset.from_tensor_slices(
    tf.strings.unicode_split(corpus, 'UTF-8')
)
ids_ds = chars_ds.map(lambda ch: table.lookup(ch), num_parallel_calls=tf.data.AUTOTUNE)
seq_ds = ids_ds.batch(SEQ_LENGTH + 1, drop_remainder=True)
def split_input_target(seq):
    return seq[:-1], seq[1:]
dataset = seq_ds.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = (
    dataset.shuffle(5_000)
           .batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE)
)
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Dataset ready: {steps_per_epoch:,} batches per epoch")

# —— 六、Transformer 模块定义 —— 
def get_positional_encoding(length, depth):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]  # 将pos设为float32
    i = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]     # 将i设为float32
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / depth)  # 分母直接使用depth的float形式
    angle_rads = pos * angle_rates
    sin = tf.sin(angle_rads[:, 0::2])
    cos = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sin, cos], axis=-1)
    return pos_encoding

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super().__init__()
        self.embed_dim = embed_dim  # 记录参数
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # 定义子层
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=None):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):  # 新增方法
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

# —— 七、模型搭建 —— 
inputs = layers.Input(shape=(SEQ_LENGTH,), dtype=tf.int64)
emb = layers.Embedding(vocab_size, EMBEDDING_DIM)(inputs)
pos_enc = get_positional_encoding(SEQ_LENGTH, EMBEDDING_DIM)
emb += pos_enc
x = emb
for _ in range(NUM_LAYERS):
    x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)
outputs = layers.Dense(vocab_size)(x)
model = Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
model.summary()

# —— 八、tqdm 进度回调 & 训练 —— 
class TqdmCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.prog = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}", unit='batch')
    def on_train_batch_end(self, batch, logs=None):
        self.prog.set_postfix(loss=f"{logs['loss']:.4f}")
        self.prog.update(1)
    def on_epoch_end(self, epoch, logs=None):
        self.prog.close()

print("Starting training...")
model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[TqdmCallback()],
    verbose=0
)

# —— 九、模型保存 —— 
model.save("pretrain_transformer.keras", save_format="keras")
print("Model saved as pretrain_transformer.keras")

