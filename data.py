from torch.utils.data import Dataset
from tools import logging
import torch
from config import AGNEWSConfig, SNLIConfig, IMDBConfig, SSTConfig
from transformers import BertTokenizer
from baseline_module.baseline_data import baseline_Tokenizer


class AGNEWS_Dataset(Dataset):
    """
    data_idx = seq,
    data_mask = mask of seq,
    label_idx = seq + 1,
    classification_label
    """
    def __init__(self, train_data=True, attack_vocab=None, debug_mode=False):
        super(AGNEWS_Dataset, self).__init__()
        if train_data:
            self.path = AGNEWSConfig.train_data_path
        else:
            self.path = AGNEWSConfig.test_data_path
        self.attack_vocab = attack_vocab
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if attack_vocab:
            self.baseline_tokenizer = baseline_Tokenizer()
        self.sen_len = AGNEWSConfig.sen_len
        self.data_tokens = []
        self.label_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            if self.attack_vocab:
                data_tokens = ['[CLS]']
                data_tokens += self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens = self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            else:
                data_tokens = ['[CLS]']
                data_tokens += self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens = self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            self.data_tokens.append(data_tokens)
            self.label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        if self.attack_vocab:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    [self.attack_vocab.get_index(token) for token in tokens])
        else:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))
                self.data_mask[i] += [0] * (self.sen_len -
                                            len(self.data_mask[i]))

            if len(self.label_idx[i]) < self.sen_len:
                self.label_idx[i] += [0] * (self.sen_len -
                                            len(self.label_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.data_mask = torch.tensor(self.data_mask)
        self.label_idx = torch.tensor(self.label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.data_mask[item], self.label_idx[
            item], self.classification_label[item]

    def __len__(self):
        return len(self.datas)


class IMDB_Dataset(Dataset):
    """
    data_idx = seq,
    data_mask = mask of seq,
    label_idx = seq + 1,
    classification_label
    """
    def __init__(self, train_data=True, attack_vocab=None, debug_mode=False):
        super(IMDB_Dataset, self).__init__()
        if train_data:
            self.path = IMDBConfig.train_data_path
        else:
            self.path = IMDBConfig.test_data_path
        self.attack_vocab = attack_vocab
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        if attack_vocab:
            self.baseline_tokenizer = baseline_Tokenizer()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sen_len = IMDBConfig.sen_len
        self.data_tokens = []
        self.label_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            if self.attack_vocab:
                data_tokens = ['[CLS]']
                data_tokens += self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens = self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            else:
                data_tokens = ['[CLS]']
                data_tokens += self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens = self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            self.data_tokens.append(data_tokens)
            self.label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        if self.attack_vocab:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    [self.attack_vocab.get_index(token) for token in tokens])
        else:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))
                self.data_mask[i] += [0] * (self.sen_len -
                                            len(self.data_mask[i]))

            if len(self.label_idx[i]) < self.sen_len:
                self.label_idx[i] += [0] * (self.sen_len -
                                            len(self.label_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.data_mask = torch.tensor(self.data_mask)
        self.label_idx = torch.tensor(self.label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.data_mask[item], self.label_idx[
            item], self.classification_label[item]

    def __len__(self):
        return len(self.datas)


class SST2_Dataset(Dataset):
    """
    data_idx = seq,
    data_mask = mask of seq,
    label_idx = seq + 1,
    classification_label
    """
    def __init__(self, train_data=True, attack_vocab=None, debug_mode=False):
        super(SST2_Dataset, self).__init__()
        if train_data:
            self.path = SSTConfig.train_data_path
        else:
            self.path = SSTConfig.test_data_path
        self.attack_vocab = attack_vocab
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        if attack_vocab:
            self.baseline_tokenizer = baseline_Tokenizer()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sen_len = IMDBConfig.sen_len
        self.data_tokens = []
        self.label_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            if self.attack_vocab:
                data_tokens = ['[CLS]']
                data_tokens += self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens = self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            else:
                data_tokens = ['[CLS]']
                data_tokens += self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens = self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            self.data_tokens.append(data_tokens)
            self.label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        if self.attack_vocab:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    [self.attack_vocab.get_index(token) for token in tokens])
        else:
            for tokens in self.label_tokens:
                self.label_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))
                self.data_mask[i] += [0] * (self.sen_len -
                                            len(self.data_mask[i]))

            if len(self.label_idx[i]) < self.sen_len:
                self.label_idx[i] += [0] * (self.sen_len -
                                            len(self.label_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.data_mask = torch.tensor(self.data_mask)
        self.label_idx = torch.tensor(self.label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.data_mask[item], self.label_idx[
            item], self.classification_label[item]

    def __len__(self):
        return len(self.datas)


class SNLI_Dataset(Dataset):
    """
    premise_data_idx = seq of premise,
    premise_data_mask = mask of premise,
    premise_data_len = len of seq_premise,
    premise_label_idx = seq_premise + 1,
    hypothesis_data_idx = seq of hypothesis,
    hypothesis_data_mask = mask of hypothesis,
    hypothesis_data_len = len of seq_hypothesis,
    hypothesis_label_idx = seq_hypothesis + 1,
    classification_label
    """
    def __init__(self, train_data=True, attack_vocab=None, debug_mode=False):
        super(SNLI_Dataset, self).__init__()
        if train_data:
            self.path = SNLIConfig.train_data_path
        else:
            self.path = SNLIConfig.test_data_path
        self.sentences_path = SNLIConfig.sentences_data_path
        self.attack_vocab = attack_vocab
        self.sentences = self.read_standard_sentences(self.sentences_path)
        self.premise_data, self.hypothesis_data, self.classification_label = self.read_standard_data(
            self.path, self.sentences, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if attack_vocab:
            self.baseline_tokenizer = baseline_Tokenizer()
        self.sen_len = SNLIConfig.sen_len
        self.premise_data_tokens = []
        self.hypothesis_data_tokens = []
        self.premise_label_tokens = []
        self.hypothesis_label_tokens = []
        self.premise_data_idx = []
        self.premise_label_idx = []
        self.premise_data_mask = []
        self.premise_data_len = []
        self.hypothesis_data_idx = []
        self.hypothesis_label_idx = []
        self.hypothesis_data_mask = []
        self.hypothesis_data_len = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def read_standard_sentences(self, path):
        sentences_id = {}
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = line.strip().split('\t')
                sentences_id[int(tokens[0].strip())] = tokens[1].strip()
        return sentences_id

    def read_standard_data(self, path, sentences, debug_mode=False):
        label_classes = SNLIConfig.label_classes
        premise_data = []
        hypothesis_data = []
        labels = []
        if debug_mode:
            i = 250
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    tokens = line.strip().split('\t')
                    labels.append(label_classes[tokens[0].strip()])
                    premise_data.append(sentences[int(tokens[1].strip())])
                    hypothesis_data.append(sentences[int(tokens[2].strip())])
                    i -= 1
                    if i == 0:
                        break
            logging(f'loading data {len(premise_data)} from {path}')
            return premise_data, hypothesis_data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = line.strip().split('\t')
                labels.append(label_classes[tokens[0].strip()])
                premise_data.append(sentences[int(tokens[1].strip())])
                hypothesis_data.append(sentences[int(tokens[2].strip())])
        logging(f'loading data {len(premise_data)} from {path}')
        return premise_data, hypothesis_data, labels

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.premise_data:
            data_tokens = ['[CLS]']
            data_tokens += self.tokenizer.tokenize(sen)[:self.sen_len - 1]
            if self.attack_vocab:
                label_tokens = self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            else:
                label_tokens = self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']

            self.premise_data_tokens.append(data_tokens)
            self.premise_label_tokens.append(label_tokens)

        for sen in self.hypothesis_data:
            data_tokens = ['[CLS]']
            data_tokens += self.tokenizer.tokenize(sen)[:self.sen_len - 1]
            if self.attack_vocab:
                label_tokens = self.baseline_tokenizer(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            else:
                label_tokens = self.tokenizer.tokenize(sen)[:self.sen_len - 1]
                label_tokens += ['[SEP]']
            self.hypothesis_data_tokens.append(data_tokens)
            self.hypothesis_label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.premise_data_tokens:
            self.premise_data_idx.append(
                self.tokenizer.convert_tokens_to_ids(tokens))
            self.premise_data_len.append(len(tokens))
            self.premise_data_mask.append([1] * len(tokens))

        if self.attack_vocab:
            for tokens in self.premise_label_tokens:
                self.premise_label_idx.append(
                    [self.attack_vocab.get_index(token) for token in tokens])
        else:
            for tokens in self.premise_label_tokens:
                self.premise_label_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.premise_data_idx)):
            if len(self.premise_data_idx[i]) < self.sen_len:
                self.premise_data_idx[i] += [0] * (
                    self.sen_len - len(self.premise_data_idx[i]))
                self.premise_data_mask[i] += [0] * (
                    self.sen_len - len(self.premise_data_mask[i]))

            if len(self.premise_label_idx[i]) < self.sen_len:
                self.premise_label_idx[i] += [0] * (
                    self.sen_len - len(self.premise_label_idx[i]))

        for tokens in self.hypothesis_data_tokens:
            self.hypothesis_data_idx.append(
                self.tokenizer.convert_tokens_to_ids(tokens))
            self.hypothesis_data_len.append(len(tokens))
            self.hypothesis_data_mask.append([1] * len(tokens))

        if self.attack_vocab:
            for tokens in self.hypothesis_label_tokens:
                self.hypothesis_label_idx.append(
                    [self.attack_vocab.get_index(token) for token in tokens])
        else:
            for tokens in self.hypothesis_label_tokens:
                self.hypothesis_label_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.hypothesis_data_idx)):
            if len(self.hypothesis_data_idx[i]) < self.sen_len:
                self.hypothesis_data_idx[i] += [0] * (
                    self.sen_len - len(self.hypothesis_data_idx[i]))
                self.hypothesis_data_mask[i] += [0] * (
                    self.sen_len - len(self.hypothesis_data_mask[i]))

            if len(self.hypothesis_label_idx[i]) < self.sen_len:
                self.hypothesis_label_idx[i] += [0] * (
                    self.sen_len - len(self.hypothesis_label_idx[i]))

    def transfor(self):
        self.premise_data_idx = torch.tensor(self.premise_data_idx)
        self.premise_data_mask = torch.tensor(self.premise_data_mask)
        self.premise_data_len = torch.tensor(self.premise_data_len)
        self.premise_label_idx = torch.tensor(self.premise_label_idx)
        self.hypothesis_data_idx = torch.tensor(self.hypothesis_data_idx)
        self.hypothesis_data_mask = torch.tensor(self.hypothesis_data_mask)
        self.hypothesis_data_len = torch.tensor(self.hypothesis_data_len)
        self.hypothesis_label_idx = torch.tensor(self.hypothesis_label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.premise_data_idx[item], self.premise_data_mask[
            item], self.premise_data_len[item], self.premise_label_idx[
                item], self.hypothesis_data_idx[
                    item], self.hypothesis_data_mask[
                        item], self.hypothesis_data_len[
                            item], self.hypothesis_label_idx[
                                item], self.classification_label[item]

    def __len__(self):
        return len(self.premise_data)


if __name__ == '__main__':
    train_dataset_orig = SNLI_Dataset(train_data=True)
    test_dataset_orig = SNLI_Dataset(train_data=False)
