import torch
import torch.nn as nn
from config import AttackConfig
from transformers import BertModel


class Seq2Seq_bert(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size=AttackConfig.hidden_size,
                 num_layers=AttackConfig.num_layers,
                 dropout=AttackConfig.dropout,
                 bidirectional=False,
                 noise_std=AttackConfig.noise_std):
        super(Seq2Seq_bert, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std
        self.bidirectional = bidirectional
        self.encoder = BertModel.from_pretrained('bert-base-uncased',
                                                 output_hidden_states=True)
        if not AttackConfig.fine_tuning:
            for param in self.encoder.parameters():
                param.requires_grad = False
        decoder_input_size = AttackConfig.hidden_size
        if AttackConfig.head_tail:
            decoder_input_size *= 2
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout)
        if bidirectional:
            self.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(self.hidden_size * 2, vocab_size))
        else:
            self.fc = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(self.hidden_size, vocab_size))

    def encode(self, inputs, inputs_mask, is_noise=False):
        """bert_based_encode

        Args:
            inputs (torch.tensor): origin input # [batch, seq_len]
            inputs_mask (torch.Tensor): origin mask # [batch, seq_len]
            is_noise (bool, optional): whether to add noise. Defaults to False.

        Returns:
            torch.tensor: hidden # [batch, seq_len, hidden_size]
        """
        encoders, pooled, all_hidden_states = self.encoder(
            inputs, attention_mask=inputs_mask)[:]
        # pooled [batch, hidden_size]
        # hidden [batch, seq_len, hidden_size]
        hidden = encoders
        state = all_hidden_states[0]
        if is_noise:
            gaussian_noise = torch.normal(mean=torch.zeros_like(hidden),
                                          std=self.noise_std)
            gaussian_noise.to(AttackConfig.train_device)
            hidden += gaussian_noise
        return hidden, state

    def decode(self, hidden, state=None, to_vocab=False):
        """lstm_based_decode
            without inputs_embedding
        Args:
            hidden (torch.tensor): bert_hidden[-1] [batch, seq_len, hidden_size]
            state (torch.tensoor): bert_hidden[0] [batch, seq_len, hidden_size]

        Returns:
            [torch.tensor]: outputs [batch, seq_len, vocab_size]
        """
        # hidden [batch, seq_len, hidden_size]
        # state [batch, seq_len, hidden_size]
        if AttackConfig.head_tail:
            # hidden [batch, seq_len, hidden_size * 2]
            hidden = torch.cat([state, hidden], 2)
        self.decoder.flatten_parameters()
        outputs, _ = self.decoder(hidden.permute(1, 0, 2))
        # outputs [batch, seq_len, vocab_size]
        if to_vocab:
            return self.fc(outputs.permute(1, 0, 2))
        return outputs.permute(1, 0, 2)

    def forward(self,
                inputs,
                inputs_mask,
                is_noise=False,
                encode_only=False,
                generator=None):
        """forward

        Args:
            inputs (torch.tensor): orginal inputs [batch, seq_len]
            inputs_mask (torch.tensor):  orginal mask [batch, seq_len]
            is_noise (bool, optional): whether add noise. Defaults to False.
            encode_only (bool, optional):  Defaults to False.
            generator (func, optional):  Defaults to None.

        Returns:
            torch.tensor: outputs [batch, seq_len, vocab_size]
        """
        hidden, state = self.encode(inputs,
                                    inputs_mask=inputs_mask,
                                    is_noise=is_noise)
        if encode_only:
            return hidden
        if not generator:
            decoded = self.decode(hidden, state)
        else:
            c_hat = generator(hidden)
            decoded = self.decode(c_hat, state)
        return self.fc(decoded)


class Generator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional=False,
                 dropout=0.3):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.linear = None
        if self.bidirectional:
            self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, inputs):
        # input: [batch_size, sen_len, hidden_size]
        out, self.hidden = self.lstm(inputs)
        if self.bidirectional:
            out_bid = self.linear(out)
            return out_bid
        # out [batch, seq_len, hidden_size]
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional=False,
                 dropout=0.3):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        if self.bidirectional:
            self.linear = nn.Linear(self.hidden_size * 2, 1)
        else:
            self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        # input: [batch_size, sen_len, hidden_size]
        out, _ = self.lstm(inputs)
        output = self.linear(out)
        return torch.mean(output.squeeze(dim=2))
