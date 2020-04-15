import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import pickle
import random
sys.path.append('../')
sys.path.append('../coco-caption/')
from pycocoevalcap.cider.cider import Cider
from data.dataset import load_dataset_cap, collate_fn_cap, get_nwords, get_nclasses
from eval.eval_cap import eval
import myopts
from models.describer_generator import Caption_generator
from models.loss import LanguageModelCriterion, ClassifierCriterion, RewardCriterion
from visualize import Visualizer
from torchnet import meter
from collections import OrderedDict
from allennlp.predictors.predictor import Predictor

def numbers_to_str(numbers):
    ret = ''
    length = len(numbers)
    is_tensor = (type(numbers) == torch.Tensor)
    for i in range(length):
        if numbers[i] == 0: break
        ret += str(numbers[i] if not is_tensor else numbers[i].item()) + ' '
    return ret.strip()

def get_self_critical_reward(model, feat0, feat1, feat_mask, pos_feat, groundtruth, probability_sample):
    batch_size = feat0.size(0)
    double_batch_size = batch_size * 2
    seq_length = probability_sample.size(1)

    greedy_sample, _ = model.sample(feat0, feat1, feat_mask, pos_feat)
    res = OrderedDict()
    gts = OrderedDict()
    greedy_sample = greedy_sample.cpu().numpy()
    probability_sample = probability_sample.cpu().numpy()

    for i in range(batch_size):
        res[i] = [numbers_to_str(probability_sample[i])]
    for i in range(batch_size, double_batch_size):
        res[i] = [numbers_to_str(greedy_sample[i - batch_size])]

    length = len(groundtruth[0])
    for i in range(batch_size):
        gts[i] = [numbers_to_str(groundtruth[i][j]) for j in range(length)]
    gts = {i:gts[i % batch_size] for i in range(double_batch_size)}
    assert len(gts.keys()) == len(res.keys()), 'len of gts.keys is not equal to that of res.keys'
    avg_cider_score, cider_score = Cider().compute_score(gts=gts, res=res)
    cider_score = np.array(cider_score)
    reward = cider_score[:batch_size] - cider_score[batch_size:]
    reward = np.repeat(reward[:, np.newaxis], seq_length, axis=1)
    return reward

def get_self_critical_textual_entailment_reward(model, feat0, feat1, feat_mask, pos_feat, groudtruth, probability_sample, opt):
    predictor = Predictor.from_path(archive_path=opt.textual_entailment_path, predictor_name='textual-entailment')

    batch_size = feat0.size(0)
    double_batch_size = batch_size * 2
    seq_length = probability_sample.size(1)

    greedy_sample, _ = model.sample(feat0, feat1, feat_mask, pos_feat)
    greedy_sample = greedy_sample.cpu().numpy()
    probability_sample = probability_sample.cpu().numpy()
    res = OrderedDict()
    gts = OrderedDict()

    for i in range(batch_size):
        res[i] = numbers_to_str(probability_sample[i])
    for i in range(batch_size, double_batch_size):
        res[i] = numbers_to_str(greedy_sample[i - batch_size])

    length = len(groudtruth[0])
    number_store = list(range(length))
    # number_store = random.sample(number_store, opt.random_select)
    for i in range(batch_size):
        gts[i] = [numbers_to_str(groudtruth[i][j]) for j in number_store]
    gts = {i: gts[i % batch_size] for i in range(double_batch_size)}
    entailment_score = np.zeros(double_batch_size)
    store = []

    for i in range(double_batch_size):
        hypothesis = res[i]
        for j in range(len(gts[i])):
            premise = gts[i][j]
            temp_dict = {'hypothesis': hypothesis, 'premise': premise}
            store.append(temp_dict)
        result = predictor.predict_batch_json(store)
        for j in range(len(result)):
            entailment_score[i] = max(entailment_score[i], result[j]['label_probs'][0])

    reward = entailment_score[:batch_size] - entailment_score[batch_size:]
    reward = np.repeat(reward[:, np.newaxis], seq_length, axis=1)
    return reward

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None: continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def train(opt):
    vis = Visualizer(env='Caption_generator')
    opt.vocab_size = get_nwords(opt.data_path)
    opt.category_size = get_nclasses(opt.data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset, test_dataset = load_dataset_cap(opt=opt)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    model = Caption_generator(opt=opt)
    infos = {}
    best_score = None
    crit = LanguageModelCriterion()
    classify_crit = ClassifierCriterion()
    rl_crit = RewardCriterion()

    if opt.start_from is not None:
        assert os.path.isdir(opt.start_from), 'opt.start_from must be a dir!'
        state_dict_path = os.path.join(opt.start_from, opt.model_name + '-bestmodel.pth')
        assert os.path.isfile(state_dict_path), 'bestmodel don\'t exist!'
        model.load_state_dict(torch.load(state_dict_path), strict=True)

        infos_path = os.path.join(opt.start_from, opt.model_name + '_infos_best.pkl')
        assert os.path.isfile(infos_path), 'infos of bestmodel don\'t exist!'
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)

        if opt.seed == 0: opt.seed = infos['opt'].seed

        best_score = infos.get('best_score', None)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    train_patience = 0
    epoch = infos.get('epoch', 0)
    loss_meter = meter.AverageValueMeter()
    eval_semantics_store = opt.eval_semantics
    is_first = True

    while True:
        if train_patience > opt.patience: break
        loss_meter.reset()
        if opt.learning_rate_decay_start != -1 and epoch > opt.learning_rate_decay_start:
            frac = int((epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every)
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            set_lr(optimizer, opt.current_lr)
        else:
            opt.current_lr = opt.learning_rate

        if opt.scheduled_sampling_start != -1 and epoch > opt.scheduled_sampling_start:
            frac = int((epoch - opt.scheduled_sampling_start) / opt.scheduled_sampling_increase_every)
            opt.sample_probability = min(opt.scheduled_sampling_increase_probability * frac, opt.scheduled_sampling_max_probability)
            model.sample_probability = opt.sample_probability

        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            sc_flag = True
            opt.eval_semantics = eval_semantics_store
            if is_first:
                best_score = None
                is_first = False
        else:
            sc_flag = False
            opt.eval_semantics = 0
        # sc_flag = True

        for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(train_loader):

            feats0 = feats0.to(device)
            feats1 = feats1.to(device)
            feat_mask = feat_mask.to(device)
            pos_feat = pos_feat.to(device)
            caps = caps.to(device)
            caps_mask = caps_mask.to(device)
            cap_classes = cap_classes.to(device)
            class_masks = class_masks.to(device)

            optimizer.zero_grad()
            if not sc_flag:
                words, categories = model(feats0, feats1, feat_mask, pos_feat, caps, caps_mask)
                loss_words = crit(words, caps, caps_mask)
                loss_cate = classify_crit(categories, cap_classes, caps_mask, class_masks)
                loss = loss_words + opt.weight_class * loss_cate
            elif not opt.eval_semantics:
                sample_dict = {}
                sample_dict.update(vars(opt))
                sample_dict.update({'sample_max':0})
                probability_sample, sample_logprobs = model.sample(feats0, feats1, feat_mask, pos_feat, sample_dict)
                reward = get_self_critical_reward(model, feats0, feats1, feat_mask, pos_feat, gts, probability_sample)
                reward = torch.from_numpy(reward).float()
                reward = reward.to(device)
                loss = rl_crit(sample_logprobs, probability_sample, reward)
            else:
                sample_dict = vars(opt)
                sample_dict.update({'sample_max':0})
                probability_sample, sample_logprobs = model.sample(feats0, feats1, feat_mask, pos_feat, sample_dict)
                reward = get_self_critical_textual_entailment_reward(model, feats0, feats1, feat_mask, pos_feat, gts, probability_sample, opt)
                reward = torch.from_numpy(reward).float()
                reward = reward.to(device)
                loss = rl_crit(sample_logprobs, probability_sample, reward)
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.detach()
            loss_meter.add(train_loss.item())

            if i % opt.visualize_every == 0:
                vis.plot('train_loss', loss_meter.value()[0])
                information = 'best_score is ' + (str(best_score) if best_score is not None else '0.0')
                information += (' reward is ' if sc_flag else ' loss is ') + str(train_loss.item())
                if sc_flag is False:
                    information += ' category_loss is ' + str(loss_cate.cpu().item())
                vis.log(information)

            is_best = False
            if (i + 1) % opt.save_checkpoint_every == 0:
                if not opt.eval_semantics or not sc_flag:
                    current_loss, current_language_state = eval(model, crit, classify_crit, valid_dataset, vars(opt))
                else:
                    current_textual_score, current_language_state = eval(model, crit, classify_crit, valid_dataset, vars(opt))
                current_score = current_language_state['CIDEr'] if not opt.eval_semantics or not sc_flag else current_textual_score
                vis.log('{}'.format('cider score is ' if not opt.eval_semantics or not sc_flag else 'textual_score is') + str(current_score))
                if best_score is None or current_score > best_score:
                    is_best = True
                    best_score = current_score
                    train_patience = 0
                else:
                    train_patience += 1

                infos['opt'] = opt
                infos['iteration'] = i
                infos['best_score'] = best_score
                infos['language_state'] = current_language_state
                infos['epoch'] = epoch
                save_state_path = os.path.join(opt.checkpoint_path, opt.model_name + '_' + str(i) + '.pth')
                torch.save(model.state_dict(), save_state_path)
                save_infos_path = os.path.join(opt.checkpoint_path, opt.model_name + '_' + 'infos_' + str(i) + '.pkl')
                with open(save_infos_path, 'wb') as f:
                    pickle.dump(infos, f)

                if is_best:
                    save_state_path = os.path.join(opt.checkpoint_path, opt.model_name + '-bestmodel.pth')
                    save_infos_path = os.path.join(opt.checkpoint_path, opt.model_name + '_' + 'infos_best.pkl')
                    torch.save(model.state_dict(), save_state_path)
                    with open(save_infos_path, 'wb') as f:
                        pickle.dump(infos, f)

        epoch += 1


if __name__ == '__main__':
    opt = myopts.parse_opt()
    train(opt=opt)