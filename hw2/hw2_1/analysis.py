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

def preprocess_video_data():
    path = '/Users/adimulamsathwika/Documents/MastersSubjects/Deep Learning/Homework 2/code_files/MLDS_hw2_1_data/'
    with open(path + 'training_label.json', 'r') as f:
        image_data = json.load(f)
    word_frequency = {}
    for data in image_data:
        for sentence in data['caption']:
            word_sentence = re.sub('[.!,;?]]', ' ', sentence).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1
    vocabulary = {}
    for word in word_frequency:
        if word_frequency[word] > 3:
            vocabulary[word] = word_frequency[word]
    special_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_to_word = {i + len(special_tokens): w for i, w in enumerate(vocabulary)}
    word_to_index = {w: i + len(special_tokens) for i, w in enumerate(vocabulary)}
    for token, index in special_tokens:
        index_to_word[index] = token
        word_to_index[token] = index 
    return index_to_word, word_to_index, vocabulary

def tokenize_input_text(input_text, vocabulary, word_to_index):
    tokenized_text = re.sub(r'[.!,;?]', ' ', input_text).split()
    for i in range(len(tokenized_text)):
        if tokenized_text[i] not in vocabulary:
            tokenized_text[i] = 3
        else:
            tokenized_text[i] = word_to_index[tokenized_text[i]]
    tokenized_text.insert(0, 1)
    tokenized_text.append(2)
    return tokenized_text

def tokenize_and_annotate_captions(caption_file, vocabulary, word_to_index):
    caption_json = '/Users/adimulamsathwika/Documents/MastersSubjects/Deep Learning/Homework 2/code_files/MLDS_hw2_1_data/' + caption_file
    annotated_captions = []
    with open(caption_json, 'r') as f:
        captions = json.load(f)
    for d in captions:
        for s in d['caption']:
            s = tokenize_input_text(s, vocabulary, word_to_index)
            annotated_captions.append((d['id'], s))
    return annotated_captions

def read_avi_files(directory_path):
    avi_dict = {}
    avi_directory = '/Users/adimulamsathwika/Documents/MastersSubjects/Deep Learning/Homework 2/code_files/MLDS_hw2_1_data/' + directory_path
    avi_files = os.listdir(avi_directory)
    for avi_file in avi_files:
        value = np.load(os.path.join(avi_directory, avi_file))
        avi_dict[avi_file.split('.npy')[0]] = value
    return avi_dict

def create_minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_list, caption_list = zip(*data)
    avi_tensor = torch.stack(avi_list, 0)
    caption_lengths = [len(cap) for cap in caption_list]
    caption_tensor = torch.zeros(len(caption_list), max(caption_lengths)).long()
    for i, cap in enumerate(caption_list):
        end = caption_lengths[i]
        caption_tensor[i, :end] = cap[:end]
    return avi_tensor, caption_tensor, caption_lengths
