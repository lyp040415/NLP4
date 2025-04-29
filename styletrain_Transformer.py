import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tqdm import tqdm
import glob

# —— 一、环境配置 ——
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# —— 二、参数设置 ——
PRETRAINED_MODEL = "pretrain_transformer_2.keras"
FINE_TUNED_MODEL = "style_transformer.keras"
SEQ_LENGTH = 100
BATCH_SIZE = 16
BUFFER_SIZE = 10000
EPOCHS = 10
CORPUS_DIR = r"D:\课程\大四\大四下\自然语言处理\第四次作业\jyxstxtqj_downcc.com"

# —— 三、自定义层定义（必须与预训练模型匹配）——
def get_positional_encoding(length, depth):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / depth)
    angle_rads = pos * angle_rates
    sin = tf.sin(angle_rads[:, 0::2])
    cos = tf.cos(angle_rads[:, 1::2])
    return tf.concat([sin, cos], axis=-1)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3, **kwargs):  # 添加**kwargs
        super().__init__(**kwargs)  # 传递参数给父类
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
    
    def get_config(self):  # 添加序列化配置
        config = super().get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.drop1.rate
        })
        return config

# —— 四、语料加载与预处理 ——
def load_jinyong_corpus(corpus_dir):
    print("Loading Jinyong novels...")
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    corpus = []
    for file in txt_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                corpus.append(f.read().replace("\r\n", "\n"))
        except UnicodeDecodeError:
            with open(file, 'r', encoding='gb18030', errors='ignore') as f:
                corpus.append(f.read().replace("\r\n", "\n"))
    return "".join(corpus)

full_text = load_jinyong_corpus(CORPUS_DIR)
print(f"Loaded {len(full_text):,} characters")

# —— 五、构建词汇映射 ——
vocab = sorted(set(full_text))
char2idx = tf.keras.layers.StringLookup(
    vocabulary=vocab, mask_token=None
)
idx2char = tf.keras.layers.StringLookup(
    vocabulary=char2idx.get_vocabulary(), invert=True, mask_token=None
)

# —— 六、创建训练数据集 ——
def text_to_ids(text):
    return char2idx(tf.strings.unicode_split(text, "UTF-8"))

text_ids = text_to_ids(full_text)
char_dataset = tf.data.Dataset.from_tensor_slices(text_ids)
sequences = char_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# —— 七、加载预训练模型 ——
print("\nLoading pre-trained model...")
model = tf.keras.models.load_model(
    PRETRAINED_MODEL,
    custom_objects={
        'TransformerBlock': TransformerBlock,
        'get_positional_encoding': get_positional_encoding
    }
)
model.summary()

# —— 八、自定义训练循环 ——
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# —— 九、执行微调训练 ——
print("\nStarting fine-tuning...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # 初始化进度条
    prog_bar = tqdm(
        dataset, 
        total=len(list(dataset.as_numpy_iterator())),
        desc="Training",
        unit="batch"
    )
    
    # 批次训练
    total_loss = 0
    for batch, (inputs, targets) in enumerate(prog_bar):
        batch_loss = train_step(inputs, targets)
        total_loss += batch_loss
        
        # 更新进度条
        prog_bar.set_postfix({
            "loss": f"{batch_loss.numpy():.4f}",
            "avg_loss": f"{total_loss/(batch+1):.4f}"
        })

# —— 十、保存微调模型 ——
model.save(FINE_TUNED_MODEL)
print(f"\nModel saved to {FINE_TUNED_MODEL}")