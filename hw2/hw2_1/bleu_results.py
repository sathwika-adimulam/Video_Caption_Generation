import sys
import torch
import json
from evaluation import test_evalulate
from train_test import TestData, Attention, EncoderRNN, Decoder, Seq2Seq
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

model = torch.load('SavedModel/model0.h5', map_location=lambda storage, loc: storage)
test_filepath = '/Users/adimulamsathwika/Documents/MastersSubjects/Deep Learning/Homework 2/code_files/MLDS_hw2_1_data/testing_data/feat'
test_dataset = TestData('{}'.format(sys.argv[1]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i2w.pickle', 'rb') as handle:
    i2w_vocab = pickle.load(handle)

model = model.cuda()
test_results = test_evaluate(test_loader, model, i2w_vocab)

with open(sys.argv[2], 'w') as f:
    for test_id, caption in test_results:
        f.write('{},{}\n'.format(test_id, caption))


test_labels = json.load(open('/Users/adimulamsathwika/Documents/MastersSubjects/Deep Learning/Homework 2/code_files/MLDS_hw2_1_data/testing_label.json'))
output_file = sys.argv[2]
results_dict = {}
with open(output_file, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma_idx = line.index(',')
        test_id = line[:comma_idx]
        caption = line[comma_idx+1:]
        results_dict[test_id] = caption

bleu_scores = []
for item in test_labels:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(results_dict[item['id']], captions, True))
    bleu_scores.append(score_per_video[0])
average_bleu = sum(bleu_scores) / len(bleu_scores)
print("Average bleu score is " + str(average_bleu))
