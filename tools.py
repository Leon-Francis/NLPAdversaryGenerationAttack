from datetime import datetime
import numpy as np


def logging(info: str):
    print('\n\r' + '[INFO]' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          '\n\r' + str(info))


def load_bert_vocab_embedding_vec(vocab_size, embeddings_size, vocab_path,
                                  embeddings_path):
    word_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as file:
        for line in file:
            word_dict[line.strip()] = len(word_dict)
    vector = np.random.normal(0.0, 0.3, [vocab_size, embeddings_size])
    with open(embeddings_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split(' ')
            word = line[0]
            if word not in word_dict:
                continue
            vector[word_dict[word]] = np.asarray(line[1:], dtype='float32')
    return vector


if __name__ == '__main__':
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(name + '.log', 'w') as f:
        f.write(1)
