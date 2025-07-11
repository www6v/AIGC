# 语义相似度计算
from sentence_transformers import SentenceTransformer, SimilarityFunction

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed some sentences
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# Change the similarity function to Manhattan distance
model.similarity_fn_name = SimilarityFunction.MANHATTAN
print(model.similarity_fn_name)
# => "manhattan"

similarities = model.similarity(embeddings, embeddings)
print(similarities)
