import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet import meter
import pickle
import os
import sys
sys.path.append('../')
import myopts
from data.dataset import load_dataset_pos, collate_fn_pos, get_nwords, get_nclasses
from eval import eval_extract
from visualize import Visualizer
from models.loss import LanguageModelCriterion, ClassifierCriterion
from models.pos_generator import Pos_generator

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None: continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def train(opt):
    vis = Visualizer(env='Pos_generator')
    opt.vocab_size = get_nwords(opt.data_path)
    opt.category_size = get_nclasses(opt.data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset, test_dataset = load_dataset_pos(opt)
    train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)
    model = Pos_generator(opt)
    infos = {}
    best_score = None
    crit = LanguageModelCriterion()
    classify_crit = ClassifierCriterion()

    if opt.start_from is not None:
        assert os.path.isdir(opt.start_from), 'opt.start_from must be a dir'
        state_dict_path = os.path.join(opt.start_from, opt.model_name + '-bestmodel.pth')
        assert os.path.isfile(state_dict_path), 'bestmodel doesn\'t exist!'
        model.load_state_dict(torch.load(state_dict_path), strict=True)

        infos_path = os.path.join(opt.start_from, opt.model_name + '_infos_best.pkl')
        assert os.path.isfile(infos_path), 'infos of bestmodel doesn\'t exist!'
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
            opt.scheduled_sample_probability = min(opt.scheduled_sampling_increase_probability * frac, opt.scheduled_sampling_max)
            model.scheduled_sample_probability = opt.scheduled_sample_probability


        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, lens, gts, video_id) in enumerate(train_loader):
            caps = caps.to(device)
            caps_mask = caps_mask.to(device)
            cap_classes = cap_classes.to(device)
            feats0 = feats0.to(device)
            feats1 = feats1.to(device)
            feat_mask = feat_mask.to(device)
            class_masks = class_masks.to(device)

            cap_classes = torch.cat([cap_classes[:, -1:], cap_classes[:, :-1]], dim=1)
            new_mask = torch.zeros_like(class_masks)

            cap_classes = cap_classes.to(device)
            new_mask = new_mask.to(device)

            for j in range(class_masks.size(0)):
                index = np.argwhere(class_masks.cpu().numpy()[j, :] != 0)[0][-1]
                new_mask[j, :index + 1] = 1.0

            optimizer.zero_grad()

            out = model(feats0, feats1, feat_mask, caps, caps_mask, cap_classes, new_mask)
            loss = classify_crit(out, cap_classes, caps_mask, class_masks)

            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.detach()

            loss_meter.add(train_loss.item())

            if (i + 1) % opt.visualize_every == 0:
                vis.plot('loss', loss_meter.value()[0])
                information = 'best_score is ' + (str(best_score) if best_score is not None else '0.0')
                information += ' current_score is ' + str(train_loss.item())
                vis.log(information)

            if (i + 1) % opt.save_checkpoint_every == 0:
                # print('i am saving!!')
                eval_kwargs = {}
                eval_kwargs.update(vars(opt))
                val_loss = eval_extract.eval_and_extract(model, classify_crit, valid_dataset, device, 'validation', eval_kwargs, False)
                # 此处输出的是交叉熵loss, 是一个正数, loss越小说明准确性越高, 为了转化为score我们需要加一个负号, score越大准确性越高
                current_score = -val_loss.cpu().item()
                is_best = False
                if best_score is None or best_score < current_score:
                    best_score = current_score
                    is_best = True
                    train_patience = 0
                else:
                    train_patience += 1
                path_of_save_model = os.path.join(opt.checkpoint_path, str(i) + '_model.pth')
                torch.save(model.state_dict(), path_of_save_model)

                infos['iteration'] = i
                infos['epoch'] = epoch
                infos['best_val_score'] = best_score
                infos['opt'] = opt

                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.model_name + '.pkl'), 'wb') as f:
                    pickle.dump(infos, f)

                if is_best:
                    path_of_save_model = os.path.join(opt.checkpoint_path, opt.model_name + '-bestmodel.pth')
                    torch.save(model.state_dict(), path_of_save_model)
                    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.model_name + '-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)


        epoch += 1




if __name__ == '__main__':
    opt = myopts.parse_opt()
    train(opt)
    pass