from config import AttackConfig
from transformers import BertTokenizer
import torch
import numpy as np


def perturb(data, Seq2Seq_model, gan_gen, baseline_model, cands_dir, refs_dir,
            attack_vocab, search_bound, samples_num):
    # Turn on evaluation mode which disables dropout.
    Seq2Seq_model.eval()
    gan_gen.eval()
    baseline_model.eval()
    with torch.no_grad():
        attack_num = 0
        attack_success_num = 0
        with open(cands_dir, "w") as f, open(refs_dir, "w") as f_1:
            for x, x_mask, y, label in data:
                x, x_mask, y, label = x.to(
                    AttackConfig.train_device), x_mask.to(
                        AttackConfig.train_device), y.to(
                            AttackConfig.train_device), label.to(
                                AttackConfig.train_device)

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # c: [batch, sen_len, hidden_size]
                z = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)

                if AttackConfig.baseline_model == 'Bert':
                    x_type = torch.zeros(y.shape, dtype=torch.int64).to(
                        AttackConfig.train_device)
                    skiped = label != baseline_model(y, x_type,
                                                     x_mask).argmax(dim=1)
                else:
                    skiped = label != baseline_model(y).argmax(dim=1)

                y = y.to(torch.device('cpu'))

                for i in range(len(y)):
                    if skiped[i].item():
                        continue

                    attack_num += 1

                    perturb_x, successed_mask = search_fast(
                        Seq2Seq_model,
                        gan_gen,
                        baseline_model,
                        label[i],
                        z[i],
                        samples_num=samples_num,
                        search_bound=search_bound)

                    if attack_vocab:
                        for stop in range(len(y[i]), 0, -1):
                            if y[i][stop - 1].item() != 0:
                                if y[i][stop - 1].item() != 2:
                                    break
                        f_1.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in y[i][:stop]
                        ]) + "\n")

                        for n, perturb_x_sample in enumerate(perturb_x):
                            if successed_mask[n].item():
                                for stop in range(len(perturb_x_sample), 0,
                                                  -1):
                                    if perturb_x_sample[stop - 1].item() != 0:
                                        if perturb_x_sample[stop -
                                                            1].item() != 2:
                                            break
                                f.write(' '.join([
                                    attack_vocab.get_word(token)
                                    for token in perturb_x_sample[:stop]
                                ]))
                                f.write('\n')
                                attack_success_num += 1
                                break
                        else:
                            for stop in range(len(perturb_x[0]), 0, -1):
                                if perturb_x[0][stop - 1].item() != 0:
                                    if perturb_x[0][stop - 1].item() != 2:
                                        break
                            f.write(' '.join([
                                attack_vocab.get_word(token)
                                for token in perturb_x[0][:stop]
                            ]))
                            f.write('\n')

                    else:
                        for stop in range(len(y[i]), 0, -1):
                            if y[i][stop - 1].item() != 0:
                                if y[i][stop - 1].item() != 102:
                                    break
                        f_1.write(
                            tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(y[i][:stop])) +
                            "\n")

                        for n, perturb_x_sample in enumerate(perturb_x):
                            if successed_mask[n].item():
                                for stop in range(len(perturb_x_sample), 0,
                                                  -1):
                                    if perturb_x_sample[stop - 1].item() != 0:
                                        if perturb_x_sample[stop -
                                                            1].item() != 102:
                                            break
                                f.write(
                                    tokenizer.convert_tokens_to_string(
                                        tokenizer.convert_ids_to_tokens(
                                            perturb_x_sample[:stop])))
                                f.write('\n')
                                attack_success_num += 1
                                break
                        else:
                            for stop in range(len(perturb_x[0]), 0, -1):
                                if perturb_x[0][stop - 1].item() != 0:
                                    if perturb_x[0][stop - 1].item() != 102:
                                        break
                            f.write(
                                tokenizer.convert_tokens_to_string(
                                    tokenizer.convert_ids_to_tokens(
                                        perturb_x[0][:stop])))
                            f.write('\n')

    return attack_success_num / attack_num


def search_fast(Seq2Seq_model, generator, baseline_model, label, z,
                samples_num, search_bound):
    # z: [sen_len, super_hidden_size]
    Seq2Seq_model.eval()
    generator.eval()
    baseline_model.eval()
    with torch.no_grad():
        dist = []
        if samples_num > 100:
            for num in range(int(samples_num / 100)):

                # search_z: [samples_num, sen_len, super_hidden_size]
                search_z = z.repeat(100, 1, 1)
                delta = torch.normal(mean=torch.zeros_like(search_z),
                                     std=search_bound)
                dist = dist + [np.sqrt(np.sum(x**2)) for x in delta.numpy()]
                delta = delta.to(AttackConfig.train_device)
                search_z += delta
                # pertub_hidden: [samples_num, sen_len, hidden_size]
                perturb_hidden = generator(search_z)
                # pertub_x: [samples_num, seq_len]
                perturb_x = Seq2Seq_model.decode(perturb_hidden,
                                                 to_vocab=True).argmax(dim=2)

                if num == 0:
                    perturb_x_all = perturb_x
                else:
                    perturb_x_all = torch.cat((perturb_x_all, perturb_x),
                                              dim=0)

                if AttackConfig.baseline_model == 'Bert':
                    perturb_x_mask = torch.ones(perturb_x.shape,
                                                dtype=torch.int64)
                    # mask before [SEP]
                    for i in range(perturb_x.shape[0]):
                        for word_idx in range(perturb_x.shape[1]):
                            if perturb_x[i][word_idx].item() == 102:
                                perturb_x_mask[i][word_idx + 1:] = 0
                                break
                    perturb_x_mask = perturb_x_mask.to(
                        AttackConfig.train_device)
                    perturb_x_type = torch.zeros(perturb_x.shape,
                                                 dtype=torch.int64).to(
                                                     AttackConfig.train_device)
                    # perturb_label: [samples_num]
                    perturb_label = baseline_model(
                        perturb_x, perturb_x_type,
                        perturb_x_mask).argmax(dim=1)
                    if num == 0:
                        perturb_label_all = perturb_label
                    else:
                        perturb_label_all = torch.cat(
                            (perturb_label_all, perturb_label), dim=0)

                else:
                    perturb_label = baseline_model(perturb_x).argmax(dim=1)
                    if num == 0:
                        perturb_label_all = perturb_label
                    else:
                        perturb_label_all = torch.cat(
                            (perturb_label_all, perturb_label), dim=0)
            else:
                dist = np.array(dist)
        else:
            # search_z: [samples_num, sen_len, super_hidden_size]
            search_z = z.repeat(samples_num, 1, 1)
            delta = torch.normal(mean=torch.zeros_like(search_z),
                                 std=search_bound).to(torch.device('cpu'))
            dist = np.array([np.sqrt(np.sum(x**2)) for x in delta.numpy()])
            delta = delta.to(AttackConfig.train_device)
            search_z += delta
            # pertub_hidden: [samples_num, sen_len, hidden_size]
            perturb_hidden = generator(search_z)
            # pertub_x: [samples_num, seq_len]
            perturb_x = Seq2Seq_model.decode(perturb_hidden,
                                             to_vocab=True).argmax(dim=2)

            if AttackConfig.baseline_model == 'Bert':
                perturb_x_mask = torch.ones(perturb_x.shape, dtype=torch.int64)
                # mask before [SEP]
                for i in range(perturb_x.shape[0]):
                    for word_idx in range(perturb_x.shape[1]):
                        if perturb_x[i][word_idx].item() == 102:
                            perturb_x_mask[i][word_idx + 1:] = 0
                            break
                perturb_x_mask = perturb_x_mask.to(AttackConfig.train_device)
                perturb_x_type = torch.zeros(perturb_x.shape,
                                             dtype=torch.int64).to(
                                                 AttackConfig.train_device)
                # perturb_label: [samples_num]
                perturb_label = baseline_model(perturb_x, perturb_x_type,
                                               perturb_x_mask).argmax(dim=1)

            else:
                perturb_label = baseline_model(perturb_x).argmax(dim=1)

            perturb_x_all = perturb_x
            perturb_label_all = perturb_label

        successed_mask = perturb_label_all != label

        sorted_perturb_x = []
        sorted_successed_mask = []
        for _, x, y in sorted(zip(dist, perturb_x_all, successed_mask),
                              key=lambda pair: pair[0]):
            sorted_perturb_x.append(x)
            sorted_successed_mask.append(y)

    return sorted_perturb_x, sorted_successed_mask
