# 必要なライブラリ
import numpy as np
from openai import OpenAI

# OpenAIクライアントの初期化
# 注意: 実際に実行する際は、以下の "your-api-key" を実際のAPIキーに置き換えてください
client = OpenAI(api_key="your-api-key")

# 分析するテキスト
TEXTS = [
    'カレー',
    'カリー',
    '高級',
]

# テキストの埋め込みを取得
response = client.embeddings.create(
    input=TEXTS,
    model="text-embedding-3-small"
)

# 埋め込みベクトルを抽出
embeddings = [item.embedding for item in response.data]

# コサイン類似度を計算する関数
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# カレーと高級のコサイン類似度
curry_luxury_similarity = cosine_similarity(embeddings[0], embeddings[2])
print(f"「カレー」と「高級」のコサイン類似度: {curry_luxury_similarity:.6f}")

# カリーと高級のコサイン類似度
karry_luxury_similarity = cosine_similarity(embeddings[1], embeddings[2])
print(f"「カリー」と「高級」のコサイン類似度: {karry_luxury_similarity:.6f}")

# 比較結果
difference = karry_luxury_similarity - curry_luxury_similarity
print(f"\n差分: {difference:.6f}")
if difference > 0:
    print("「カリー」の方が「高級」との類似度が高いです")
else:
    print("「カレー」の方が「高級」との類似度が高いです")

# 参考：カレーとカリーの類似度
curry_karry_similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"\n「カレー」と「カリー」のコサイン類似度: {curry_karry_similarity:.6f}")
