from gensim.models import Word2Vec

# 输入语料
sentences = [["I", "love", "programming"], ["I", "hate", "bugs"], ["Python", "is", "awesome"]]
# 构建 Word2Vec 模型
model = Word2Vec(sentences, vector_size=3, window=2, min_count=1, workers=4)
# 获取 "love" 的词向量
vector = model.wv['love']
print(f"Vector for 'love': {vector}")
# 输出相似词
similar_words = model.wv.most_similar('love', topn=3)
print(f"Similar words to 'love': {similar_words}")
