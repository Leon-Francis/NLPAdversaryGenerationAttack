import torch

config_path = './config.py'
bert_vocab_size = 30522


class AttackConfig():
    output_dir = r'./output'
    train_multi_cuda = False
    cuda_idx = 0
    if train_multi_cuda:
        multi_cuda_idx = [0, 1]
        cuda_idx = multi_cuda_idx[0]
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'IMDB'  # choices = 'IMDB', 'AGNEWS', 'SNLI'
    baseline_model = 'Bert'  # choices = 'LSTM', 'TextCNN', 'BidLSTM', 'Bert'
    debug_mode = False
    epochs = 30
    batch_size = 16

    load_pretrained_Seq2Seq = True
    head_tail = False
    Seq2Seq_BidLSTM = False
    fine_tuning = False

    if fine_tuning:
        Seq2Seq_learning_rate_BERT = 5e-6
        Seq2Seq_learning_rate_LSTM = 1e-4
    else:
        Seq2Seq_learning_rate = 1e-3
    gen_learning_rate = 1e-3
    dis_learning_rate = 1e-3

    hidden_size = 768
    num_layers = 3
    dropout = 0.3
    vocab_size = bert_vocab_size
    gan_clamp = 0.01
    noise_std = 0.2
    samples_num = 20

    gan_schedule = [2, 4, 6]
    seq2seq_train_times = 1
    gan_gen_train_times = 1
    gan_dis_train_times = 5

    if load_pretrained_Seq2Seq:
        if dataset == 'IMDB':
            if baseline_model == 'Bert':
                pretrained_Seq2Seq_path = r'./output/seq2seq_model/IMDB/1619438388/Seq2Seq_model.pt'


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclImdb/train.std'
    test_data_path = r'./dataset/IMDB/aclImdb/test.std'
    labels_num = 2
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 230
    vocab_size = bert_vocab_size


class SSTConfig():
    train_data_path = r'./dataset/SST2/train.std'
    test_data_path = r'./dataset/SST2/test.std'
    labels_num = 2
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 20
    vocab_size = bert_vocab_size


class AGNEWSConfig():
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    labels_num = 4
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 50
    vocab_size = bert_vocab_size


class SNLIConfig():
    train_data_path = r'./dataset/SNLI/train.txt'
    test_data_path = r'./dataset/SNLI/test.txt'
    sentences_data_path = r'./dataset/SNLI/sentences.txt'
    labels_num = 3
    label_classes = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 15
    vocab_size = bert_vocab_size
