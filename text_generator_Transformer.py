import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tqdm import tqdm
import glob

SEQ_LENGTH = 100
CORPUS_DIR = r"D:\课程\大四\大四下\自然语言处理\第四次作业\jyxstxtqj_downcc.com"
MODEL_PATH = r"D:\课程\大四\大四下\自然语言处理\第四次作业\style_transformer_2.keras"

def get_positional_encoding(length, depth):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / depth)
    angle_rads = pos * angle_rates
    sin = tf.sin(angle_rads[:, 0::2])
    cos = tf.cos(angle_rads[:, 1::2])
    return tf.concat([sin, cos], axis=-1)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 显式记录参数
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        # 延迟构建确保形状正确
        self.att = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.drop1 = None
        self.drop2 = None

    def build(self, input_shape):
        # 确保在build阶段创建可训练权重
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(self.embed_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(self.rate)
        self.drop2 = tf.keras.layers.Dropout(self.rate)
        super().build(input_shape)

    def call(self, x, training=None):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

# —— 修正3：改进生成函数处理短输入 ——
def transformer_generate(model, start_str, num_generate=500, temp=1.0):
    # 转换输入为ID序列
    input_chars = tf.strings.unicode_split([start_str], 'UTF-8').numpy()[0]
    input_ids = char2idx(input_chars)[tf.newaxis, :]
    
    # 填充短于SEQ_LENGTH的输入
    if input_ids.shape[1] < SEQ_LENGTH:
        pad_len = SEQ_LENGTH - input_ids.shape[1]
        input_ids = tf.pad(input_ids, [[0,0], [pad_len,0]], constant_values=0)
    else:
        input_ids = input_ids[:, -SEQ_LENGTH:]
    
    generated = []
    for _ in tqdm(range(num_generate), desc="生成进度"):
        # 确保输入长度始终为SEQ_LENGTH
        model_input = input_ids[:, -SEQ_LENGTH:]
        
        # 获取预测结果
        predictions = model(model_input)
        
        # 提取最后一个时间步
        last_step = predictions[:, -1, :]
        scaled_logits = last_step / temp
        
        # 采样下一个字符
        sampled_id = tf.random.categorical(scaled_logits, num_samples=1)
        decoded_char = idx2char(sampled_id).numpy()[0][0].decode('utf-8')
        
        # 更新输入序列（保持固定长度）
        input_ids = tf.concat([input_ids, sampled_id], axis=-1)[:, -SEQ_LENGTH-1:]
        generated.append(decoded_char)
    
    return start_str + ''.join(generated)

# —— 语料加载与模型加载 ——
def load_jinyong_corpus(corpus_dir):
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
vocab = sorted(set(full_text))

char2idx = tf.keras.layers.StringLookup(
    vocabulary=vocab, mask_token=None)
idx2char = tf.keras.layers.StringLookup(
    vocabulary=char2idx.get_vocabulary(), invert=True, mask_token=None)

print("\nLoading style model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'TransformerBlock': TransformerBlock,
        'get_positional_encoding': get_positional_encoding
    }
)

# 生成示例
generated_text = transformer_generate(
    model,
    start_str="郭靖对黄蓉说",
    num_generate=1000,
    temp=0.7
)

print("\n生成结果:\n" + "="*60)
print(generated_text)
print("="*60)

output_path = os.path.join(os.path.dirname(MODEL_PATH), "transformer_text.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(generated_text)
print(f"\n文本已保存至: {output_path}")