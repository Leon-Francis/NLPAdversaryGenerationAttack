import torch
from torch import nn, optim
import os
from config import AttackConfig, config_path
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tools import logging
from datetime import datetime
from shutil import copyfile
from data import AGNEWS_Dataset, IMDB_Dataset, SNLI_Dataset, SST2_Dataset
from transformers import BertTokenizer
from baseline_module.baseline_model_builder import BaselineModelBuilder
from gan_model import Seq2Seq_bert, Generator, Discriminator
from perturb import perturb
from calc_BertScore_ppl import calc_bert_score_ppl


def train_Seq2Seq(train_data, Seq2Seq_model, criterion, optimizer):
    Seq2Seq_model.train()
    Seq2Seq_model.zero_grad()
    x, x_mask, y, _ = train_data
    x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
        AttackConfig.train_device), y.to(AttackConfig.train_device)
    logits = Seq2Seq_model(x, x_mask, is_noise=False)
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    return loss.item()


def train_gan_d(train_data, Seq2Seq_model, gan_gen, gan_dis, optimizer_gan_d):
    Seq2Seq_model.train()
    gan_gen.train()
    gan_dis.train()
    gan_dis.zero_grad()

    x, x_mask, y, _ = train_data
    # positive samples ----------------------------
    # generate real codes
    real_hidden = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)

    # real loss
    errD_real = gan_dis(real_hidden)

    # negative samples ----------------------------
    # generate fake codes
    fake_hidden = gan_gen(
        Seq2Seq_model(x, x_mask, is_noise=True, encode_only=True))

    # fake loss
    errD_fake = gan_dis(fake_hidden).detach()

    loss = -errD_real + errD_fake
    loss.backward()

    optimizer_gan_d.step()

    for p in gan_dis.parameters():
        p.data.clamp_(-AttackConfig.gan_clamp, AttackConfig.gan_clamp)

    return loss.item()


def train_gan_g(train_data, Seq2Seq_model, gan_gen, gan_dis, baseline_model,
                optimizer_gan_g, criterion_ce):
    Seq2Seq_model.train()
    gan_gen.train()
    gan_dis.train()
    baseline_model.train()
    gan_gen.zero_grad()

    x, x_mask, y, label = train_data

    fake_hidden = gan_gen(
        Seq2Seq_model(x, x_mask, is_noise=True, encode_only=True))

    # similarty loss
    loss_S = gan_dis(fake_hidden)

    # attack loss
    perturb_x_logits = Seq2Seq_model.decode(fake_hidden)
    perturb_logits = baseline_model.forward_with_embedding(
        F.softmax(perturb_x_logits, dim=2))

    loss_A = criterion_ce(perturb_logits, (label + 1) % 2)
    # loss / backprop

    loss = loss_A - loss_S
    loss.backward()

    optimizer_gan_g.step()

    return loss.item()


def evaluate_gan(test_data, Seq2Seq_model, gan_gen, dir, attack_vocab):
    Seq2Seq_model.eval()
    gan_gen.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging(f'Saving evaluate of gan outputs into {dir}')
    with torch.no_grad():

        for x, x_mask, y, _ in test_data:
            x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(AttackConfig.train_device)

            # sentence -> encoder -> decoder
            Seq2Seq_outputs = Seq2Seq_model(x, x_mask, is_noise=False)
            # Seq2Seq_idx: [batch, seq_len]
            Seq2Seq_idx = Seq2Seq_outputs.argmax(dim=2)

            # sentence -> encoder -> generator ->  decoder
            # eagd_outputs: [batch, seq_len, vocab_size]
            eagd_outputs = Seq2Seq_model(x,
                                         x_mask,
                                         is_noise=True,
                                         generator=gan_gen)
            # eagd_idx: [batch_size, sen_len]
            eagd_idx = eagd_outputs.argmax(dim=2)

            if attack_vocab:
                with open(dir, 'a') as f:
                    for i in range(len(y)):
                        f.write('------orginal sentence---------\n')
                        f.write(' '.join(
                            [attack_vocab.get_word(token)
                             for token in y[i]]) + '\n')
                        f.write('------setence -> encoder -> decoder-------\n')
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in Seq2Seq_idx[i]
                        ]) + '\n')
                        f.write(
                            '------sentence -> encoder -> inverter -> generator -> decoder-------\n'
                        )
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in eagd_idx[i]
                        ]) + '\n' * 2)
            else:
                with open(dir, 'a') as f:
                    for i in range(len(y)):
                        f.write('------orginal sentence---------\n')
                        f.write(
                            tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(y[i])) + '\n')
                        f.write('------setence -> encoder -> decoder-------\n')
                        f.write(
                            tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(
                                    Seq2Seq_idx[i])) + '\n')
                        f.write(
                            '------sentence -> encoder -> inverter -> generator -> decoder-------\n'
                        )
                        f.write(
                            tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(eagd_idx[i])) +
                            '\n' * 2)


def save_all_models(Seq2Seq_model, gan_gen, gan_dis, dir):
    logging('Saving models...')
    torch.save(Seq2Seq_model.state_dict(), dir + '/Seq2Seq_model.pt')
    torch.save(gan_gen.state_dict(), dir + '/gan_gen.pt')
    torch.save(gan_dis.state_dict(), dir + '/gan_dis.pt')


def build_dataset(attack_vocab):
    if AttackConfig.dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            attack_vocab=attack_vocab,
                                            debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           attack_vocab=attack_vocab,
                                           debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'SST2':
        train_dataset_orig = SST2_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = SST2_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=AttackConfig.debug_mode)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=AttackConfig.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=AttackConfig.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def save_config(path):
    copyfile(config_path, path + r'/config.txt')


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(AttackConfig.cuda_idx))

    cur_dir = AttackConfig.output_dir + '/gan_model/' + AttackConfig.dataset + '/' + AttackConfig.baseline_model + '/' + datetime.now(
    ).strftime("%Y-%m-%d_%H:%M:%S")
    cur_dir_models = cur_dir + '/models'
    # make output directory if it doesn't already exist
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_dir_models)
    logging('Saving into directory' + cur_dir)
    save_config(cur_dir)

    baseline_model_builder = BaselineModelBuilder(AttackConfig.dataset,
                                                  AttackConfig.baseline_model,
                                                  AttackConfig.train_device,
                                                  is_load=True)

    # prepare dataset
    logging('preparing data...')
    train_data, test_data = build_dataset(baseline_model_builder.vocab)

    # init models
    logging('init models, optimizer, criterion...')
    Seq2Seq_model = Seq2Seq_bert(
        baseline_model_builder.vocab.num
        if baseline_model_builder.vocab else AttackConfig.vocab_size,
        bidirectional=AttackConfig.Seq2Seq_BidLSTM).to(
            AttackConfig.train_device)
    if AttackConfig.load_pretrained_Seq2Seq:
        Seq2Seq_model.load_state_dict(
            torch.load(AttackConfig.pretrained_Seq2Seq_path,
                       map_location=AttackConfig.train_device))
    gan_gen = Generator(AttackConfig.hidden_size, AttackConfig.hidden_size,
                        AttackConfig.num_layers).to(AttackConfig.train_device)
    gan_dis = Discriminator(AttackConfig.hidden_size, AttackConfig.hidden_size,
                            AttackConfig.num_layers).to(
                                AttackConfig.train_device)
    baseline_model = baseline_model_builder.net

    if AttackConfig.fine_tuning:
        optimizer_Seq2Seq = optim.AdamW(
            [{
                'params': Seq2Seq_model.encoder.parameters(),
                'lr': AttackConfig.Seq2Seq_learning_rate_BERT
            }, {
                'params': Seq2Seq_model.decoder.parameters()
            }, {
                'params': Seq2Seq_model.fc.parameters()
            }],
            lr=AttackConfig.Seq2Seq_learning_rate_LSTM)
    else:
        optimizer_Seq2Seq = optim.AdamW(Seq2Seq_model.parameters(),
                                        lr=AttackConfig.Seq2Seq_learning_rate)
    optimizer_gan_gen = optim.RMSprop(gan_gen.parameters(),
                                      lr=AttackConfig.gen_learning_rate)
    optimizer_gan_dis = optim.RMSprop(gan_dis.parameters(),
                                      lr=AttackConfig.dis_learning_rate)
    # init criterion
    criterion_ce = nn.CrossEntropyLoss().to(AttackConfig.train_device)

    logging('Training Model...')

    niter_gan = 1

    for epoch in range(AttackConfig.epochs):
        if epoch in AttackConfig.gan_schedule:
            niter_gan += 1
        niter = 0
        total_loss_Seq2Seq = 0
        total_loss_gan_g = 0
        total_loss_gan_d = 0
        logging(f'Training {epoch} epoch')
        for x, x_mask, y, label in train_data:
            niter += 1
            x, x_mask, y, label = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(
                    AttackConfig.train_device), label.to(
                        AttackConfig.train_device)

            if not AttackConfig.load_pretrained_Seq2Seq:
                for i in range(AttackConfig.seq2seq_train_times):
                    total_loss_Seq2Seq += train_Seq2Seq(
                        (x, x_mask, y, label), Seq2Seq_model, criterion_ce,
                        optimizer_Seq2Seq)
            else:
                if AttackConfig.fine_tuning:
                    for i in range(AttackConfig.seq2seq_train_times):
                        total_loss_Seq2Seq += train_Seq2Seq(
                            (x, x_mask, y, label), Seq2Seq_model, criterion_ce,
                            optimizer_Seq2Seq)

            for k in range(niter_gan):
                for i in range(AttackConfig.gan_dis_train_times):
                    total_loss_gan_d += train_gan_d(
                        (x, x_mask, y, label), Seq2Seq_model, gan_gen, gan_dis,
                        optimizer_gan_dis)

                for i in range(AttackConfig.gan_gen_train_times):
                    total_loss_gan_g += train_gan_g(
                        (x, x_mask, y, label), Seq2Seq_model, gan_gen, gan_dis,
                        baseline_model, optimizer_gan_gen, criterion_ce)

            if niter % 100 == 0:
                # decaying noise
                logging(
                    f'epoch {epoch}, niter {niter}:Loss_Seq2Seq: {total_loss_Seq2Seq / niter / AttackConfig.batch_size / AttackConfig.seq2seq_train_times}, Loss_gan_g: {total_loss_gan_g / niter / AttackConfig.batch_size / AttackConfig.gan_gen_train_times}, Loss_gan_a: {total_loss_gan_d / niter / AttackConfig.batch_size / AttackConfig.gan_dis_train_times}'
                )

        # end of epoch --------------------------------
        # evaluation

        logging(f'epoch {epoch} evaluate gan')
        evaluate_gan(test_data, Seq2Seq_model, gan_gen,
                     cur_dir_models + f'/epoch{epoch}_evaluate_gan',
                     baseline_model_builder.vocab)

        if (epoch + 1) % 5 == 1:  # **
            os.makedirs(cur_dir_models + f'/epoch{epoch}')
            save_all_models(Seq2Seq_model, gan_gen, gan_dis,
                            cur_dir_models + f'/epoch{epoch}')

            logging(f'epoch {epoch} Staring perturb')
            # attach_acc: [search_time, sample_num]
            output_dir = f'./texts/OUR/{AttackConfig.dataset}/{AttackConfig.baseline_model_name}/epoch{epoch+1}'
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            refs_dir = output_dir + f'/refs_{AttackConfig.noise_std}_{AttackConfig.samples_num}.txt'
            cands_dir = output_dir + f'/cands_{AttackConfig.noise_std}_{AttackConfig.samples_num}.txt'
            attack_acc = perturb(test_data, Seq2Seq_model, gan_gen,
                                 baseline_model_builder.net, cands_dir,
                                 refs_dir, baseline_model_builder.vocab,
                                 AttackConfig.noise_std,
                                 AttackConfig.samples_num)
            logging(
                f'search_bound={AttackConfig.noise_std}, sample={AttackConfig.samples_num}'
            )
            logging(f'attack_acc={attack_acc}')
            ppl, bert_score = calc_bert_score_ppl(cands_dir, refs_dir,
                                                  AttackConfig.train_device)

            logging(f'ppl={ppl}')
            logging(f'bert_score={bert_score}')
