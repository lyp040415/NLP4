import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 关闭 oneDNN 警告
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像

from transformers import pipeline

# 指定 TensorFlow 框架
generator = pipeline(
    "text-generation",
    model="uer/gpt2-chinese-cluecorpussmall",
    framework="tf"
)

# 生成文本
result = generator("郭靖对黄蓉说，", max_length=1000)
print(result[0]['generated_text'])