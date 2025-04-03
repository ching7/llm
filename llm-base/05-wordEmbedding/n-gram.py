from collections import defaultdict
import random

# 训练文本
text = "I love this product very much. I love this shop. I hate bad products."

# 分词
words = text.split()


# 创建 n-gram 模型（支持 Unigram, Bigram, Trigram, 4-gram）
def create_ngram_model(words, n):
    ngram_model = defaultdict(list)
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])  # 生成 n-gram
        ngram_model[ngram[:-1]].append(ngram[-1])  # n-gram 的前部分作为键，最后一个单词作为值
    return ngram_model


# 拉普拉斯平滑：词汇表大小
V = len(set(words))  # 词汇表大小


# 计算 n-gram 概率
def ngram_probability(model, ngram_prefix, next_word):
    count_ngram = model[ngram_prefix].count(next_word)
    count_prefix = len(model[ngram_prefix])
    return (count_ngram + 1) / (count_prefix + V)


# 生成 n-gram 文本
def generate_ngram_text(model, n, start_word, length=10):
    # 其他 n-gram 模型：通过 n-gram 前缀生成下一个词
    current_ngram = tuple([start_word] * (n - 1))  # 根据模型大小初始化
    result = [start_word]
    for _ in range(length):
        if model[current_ngram]:
            next_word = random.choices(model[current_ngram],
                                       weights=[ngram_probability(model, current_ngram, word) for word in
                                                model[current_ngram]])[0]
            result.append(next_word)
            current_ngram = tuple(list(current_ngram[1:]) + [next_word])  # 滚动更新 n-gram
    return ' '.join(result)


# 创建 Unigram, Bigram, Trigram 和 4-gram 模型
unigram_model = create_ngram_model(words, 1)
bigram_model = create_ngram_model(words, 2)
trigram_model = create_ngram_model(words, 3)
fourgram_model = create_ngram_model(words, 4)

# 生成不同 n-gram 文本
print("Generated Sentence with Unigram Model:")
print(generate_ngram_text(unigram_model, 1, "I"))

print("\nGenerated Sentence with Bigram Model:")
print(generate_ngram_text(bigram_model, 2, "I"))

print("\nGenerated Sentence with Trigram Model:")
print(generate_ngram_text(trigram_model, 3, "I"))

print("\nGenerated Sentence with 4-gram Model:")
print(generate_ngram_text(fourgram_model, 4, "I"))
