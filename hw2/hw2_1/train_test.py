import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset
from analysis import tokenize_and_annotate_captions, read_avi_files

class TrainingDataset(Dataset):
    def __init__(self, label_file_path, avi_files_dir, word_dict, word_to_index_dict):
        self.label_file_path = label_file_path
        self.avi_files_dir = avi_files_dir
        self.word_dict = word_dict
        self.avi_data = read_avi_files(label_file_path)
        self.word_to_index_dict = word_to_index_dict
        self.data_pairs = tokenize_and_annotate_captions(avi_files_dir, word_dict, word_to_index_dict)
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pairs[idx]
        avi_data = torch.Tensor(self.avi_data[avi_file_name])
        avi_data += torch.Tensor(avi_data.size()).random_(0, 2000) / 10000.
        return torch.Tensor(avi_data), torch.Tensor(sentence)

class TestData(Dataset):
    def __init__(self, data_path):
        self.video_data = []
        files = os.listdir(data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(data_path, file))
            self.video_data.append([key, value])
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        return self.video_data[idx]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.weight_linear = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_t, enc_outputs):
        batch_size, seq_len, feature_dim = enc_outputs.size()
        h_t = h_t.view(batch_size, 1, feature_dim).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((enc_outputs, h_t), 2).view(-1, 2 * self.hidden_dim)
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attn_weights = self.weight_linear(x)
        attn_weights = attn_weights.view(batch_size, seq_len)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.compression = nn.Linear(4096, 512)
        self.dropout_layer = nn.Dropout(0.3)
        self.gru_layer = nn.GRU(512, 512, batch_first=True)

    def forward(self, input_tensor):
        batch_size, seq_len, feat_dim = input_tensor.size()    
        input_tensor = input_tensor.view(-1, feat_dim)
        input_tensor = self.compression(input_tensor)
        input_tensor = self.dropout_layer(input_tensor)
        input_tensor = input_tensor.view(batch_size, seq_len, 512)
        output_tensor, hidden_state = self.gru_layer(input_tensor)
        return output_tensor, hidden_state

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_embedding_dim, dropout_prob=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(dropout_prob)
        self.gru = nn.GRU(hidden_size + word_embedding_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def decode(self, encoder_last_hidden_state, encoder_output, target=None, mode='train', training_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_log_probs = []
        seq_predictions = []
        target = self.embedding(target)
        _, seq_len, _ = target.size()
        for i in range(seq_len-1):
            teacher_forcing_ratio = self.get_teacher_forcing_ratio(training_steps=training_steps)
            if random.uniform(0.05, 0.995) > teacher_forcing_ratio:
                current_input_word = target[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            log_probs = self.to_final_output(gru_output.squeeze(1))
            seq_log_probs.append(log_probs.unsqueeze(1))
            decoder_current_input_word = log_probs.unsqueeze(1).max(2)[1]
        seq_log_probs = torch.cat(seq_log_probs, dim=1)
        seq_predictions = seq_log_probs.max(2)[1]
        return seq_log_probs, seq_predictions
        
    def generate(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.cuda()
        seq_log_probs = []
        seq_predictions = []
        assumed_seq_len = 28
        for i in range(assumed_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            log_probs = self.to_final_output(gru_output.squeeze(1))
            seq_log_probs.append(log_probs.unsqueeze(1))
            decoder_current_input_word = log_probs.unsqueeze(1).max(2)[1]
        seq_log_probs = torch.cat(seq_log_probs, dim=1)
        seq_predictions = seq_log_probs.max(2)[1]
        return seq_log_probs, seq_predictions

    def get_teacher_forcing_ratio(self, training_steps):
        return (expit(training_step/20 +0.85))

class Seq2Seq(nn.Module):
    def __init__(self, encoder_model, decoder_model):
        super(Seq2Seq, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
    
    def forward(self, input_feature, mode, target_seq=None, teacher_forcing_ratio=None):
        encoder_output, encoder_last_hidden_state = self.encoder_model(input_feature)
        if mode == 'train':
            seq_log_prob, seq_predictions = self.decoder_model(encoder_last_hidden=encoder_last_hidden_state, encoder_output=encoder_output,
                                                               targets=target_seq, mode=mode, tr_steps=teacher_forcing_ratio)
        elif mode == 'inference':
            seq_log_prob, seq_predictions = self.decoder_model.generate(encoder_last_hidden=encoder_last_hidden_state, encoder_output=encoder_output)
        return seq_log_prob, seq_predictions
