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
from analysis import preprocess_video_data
from train_test import EncoderRNN, Decoder, Seq2Seq

def compute_loss(loss_fn, predictions, targets, lengths):
    batch_size = len(predictions)
    predictions_cat = None
    targets_cat = None
    flag = True
    for batch in range(batch_size):
        predict = predictions[batch]
        ground_truth = targets[batch]
        seq_len = lengths[batch] - 1
        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predictions_cat = predict
            targets_cat = ground_truth
            flag = False
        else:
            predictions_cat = torch.cat((predictions_cat, predict), dim=0)
            targets_cat = torch.cat((targets_cat, ground_truth), dim=0)
    loss = loss_fn(predictions_cat, targets_cat)
    avg_loss = loss / batch_size
    return loss

def train_model(model, epoch_num, loss_func, params, optimizer, train_data_loader):
    model.train()
    print(epoch_num)
    for batch_idx, batch in enumerate(train_data_loader):
        video_features, ground_truths, seq_lengths = batch
        video_features, ground_truths = video_features.cuda(), ground_truths.cuda()
        video_features, ground_truths = Variable(video_features), Variable(ground_truths)
        optimizer.zero_grad()
        seq_log_probs, seq_predictions = model(video_features, target_sentences=ground_truths, mode='train', tr_steps=epoch_num)
        ground_truths = ground_truths[:, 1:]
        loss = compute_loss(loss_func, seq_log_probs, ground_truths, seq_lengths)
        loss.backward()
        optimizer.step()
    loss_value = loss.item()
    print(loss_value)

def test_evalulate(test_loader, model, index_to_word):
    model.eval()
    results = []
    for idx, data in enumerate(test_loader):
        video_id, video_features = data
        video_features = video_features.cuda()
        video_id, video_features = video_id, Variable(video_features).float()
        sequence_log_prob, sequence_predictions = model(video_features, mode='inference')
        test_predictions = sequence_predictions
        result = [[index_to_word[x.item()] if index_to_word[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        results_zip = zip(video_id, result)
        for r in results_zip:
            results.append(r)
    return results

def main():
    idx_to_word, word_to_idx, word_dict = preprocess_video_data()
    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(idx_to_word, handle, protocol = pickle.HIGHEST_PROTOCOL)
    label_file = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = training_data(label_file, files_dir, word_dict, word_to_idx)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    epochs_n = 100
    encoder = EncoderRNN()
    decoder = Decoder(512, len(idx_to_word) +4, len(idx_to_word) +4, 1024, 0.3)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.001)
    for epoch in range(epochs_n):
        train_model(model, epoch+1, loss_fn, parameters, optimizer, train_dataloader) 
    torch.save(model, "{}/{}.h5".format('SavedModel', 'model0'))
    print("Training is finished")

    
if __name__ == "__main__":
    main()