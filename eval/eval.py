import torch
import torch.nn
import numpy as np
import sys
from collections import OrderedDict
sys.path.append('../')
sys.path.append('../coco-caption/')
from data.dataset import load_dataset_cap, collate_fn_cap, get_itow
from models.describer_generator import Caption_generator
from models.loss import ClassifierCriterion, LanguageModelCriterion, RewardCriterion
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader

def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    gts = OrderedDict()
    for i in range(len(gts)):
        gts[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]

    sampled = {i: sample_seqs[i] for i in range(len(sample_seqs))}
    gts = {i: gts[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(gts, sampled)
    avg_cider_score, cider_score = Cider().compute_score(gts, sampled)
    avg_meteor_score, meteor_score = Meteor().compute_score(gts, sampled)
    avg_rouge_score, rouge_score = Rouge().compute_score(gts, sampled)

    # print('BLEU1:{}\nBLEU2:{}\nBLEU3:{}\nBLEU4:{}\nMETEOR:{}\nROUGE:{}CIDEr:{}\n'.format(avg_bleu_score[0],
    #                                                                                      avg_bleu_score[1],
    #                                                                                      avg_bleu_score[2],
    #                                                                                      avg_bleu_score[3],
    #                                                                                      avg_meteor_score,
    #                                                                                      avg_rouge_score,
    #                                                                                      avg_cider_score))
    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}

def decode_idx(seqs, itow):
    length, seq_len = seqs.size(0), seqs.size(1)
    ret = [[] for i in range(length)]
    for i in range(length):
        for j in range(seq_len):
            ret[i].append(itow[seqs[i][j]])
    return ret

def eval(model, crit, classify_crit, dataset, eval_kwargs={}):
    # lang_eval = eval_kwargs.get('lang_eval', 1)
    data_path = eval_kwargs.get('data_path', None)
    batch_size = eval_kwargs.get('batch_size', 64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert data_path is not None, 'The data_path is not exist!'

    model.eval()
    loss_sum = 0
    loss_number = 1e-8
    total_prediction = []
    total_groundtruth = []
    id_word = get_itow(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cap)
    for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(data_loader):
        feats0 = feats0.to(device)
        feats1 = feats1.to(device)
        feat_mask = feat_mask.to(device)
        pos_feat = pos_feat.to(device)
        caps = caps.to(device)
        caps_mask = caps_mask.to(device)
        cap_classes = cap_classes.to(device)
        class_masks = class_masks.to(device)

        seq, seq_probabilities = model.sample(feats0, feats1, feat_mask, pos_feat)
        words, category = model(feats0, feats1, feat_mask, pos_feat, caps, caps_mask)
        classify_loss = classify_crit(category, cap_classes, caps_mask, class_masks)
        language_loss = crit(seq_probabilities, caps, caps_mask)
        classify_loss = classify_loss.detach().cpu().numpy()
        language_loss = language_loss.detach().cpu().numpy()
        loss_sum += (classify_loss + language_loss)
        loss_number += 1
        seq = seq.cpu().numpy()
        gts = gts.cpu().numpy()
        for t in range(batch_size):
            total_prediction.append(seq[t])
            temp = []
            number = len(gts[t])
            for x in range(number):
                temp.append(gts[t][x])
            total_groundtruth.append(temp)

    language_state = language_eval(total_prediction, total_groundtruth)
    sentence = decode_idx(np.array(total_prediction[:10]), id_word)
    print('######take a look at consequence#######')
    print(sentence)

    return loss_sum / loss_number, language_state

if __name__ == '__main__':
    import myopts
    from models.describer_generator import Caption_generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = myopts.parse_opt()
    train_dataset, valid_dataset, test_dataset = load_dataset_cap(opt)
    model = Caption_generator(opt)
    model = model.to(device)
    classify_crit = ClassifierCriterion()
    crit = LanguageModelCriterion()
    eval_kwargs = {}
    eval_kwargs.update(vars(opt))
    avg_loss, language_state = eval(model, crit, classify_crit, valid_dataset, eval_kwargs)
    print('avg_loss is ', avg_loss)
    print('language_state is ', language_state)
