import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchnet import meter
import pickle
import time
import os
import sys
sys.path.append('../')
from myopts import *
from data.dataset import *
from eval import eval_extract
from visualize import Visualizer
from models.loss import *
from models.pos_generator import *

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None: continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def train(opt, device):
    vis = Visualizer(env='picture of loss')
    opt.vocab_size = get_nwords(opt.data_path)
    opt.category_size = get_nclasses(opt.data_path)
    train_dataset, valid_dataset, test_dataset = load_dataset_pos(opt)
    infos = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_' + opt.model_name + '-best.pkl')) as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ['rnn_size', 'num_layers']
            for check in need_be_same:
                assert vars(saved_model_opt)[check] == vars(opt)[check], 'command line argument and saved model disagree on %s' % (check)

        if opt.seed == 0:
            opt.seed = infos['opt'].seed

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    else:
        best_val_score = None

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = pos_generator(opt=opt)
    if opt.start_from is not None:
        assert os.path.isfile(opt.start_from + opt.model_name + '-bestmodel.pth'), 'bestmodel saving is not exist!'
        model.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_name + '-bestmodel.pth')), strict=False)
    model.to(device)
    # model.cuda()
    model.train()
    # print('######model.device is ', type(model), ' ######')

    crit = LanguageModelCriterion()
    classify_crit = ClassifierCriterion()
    rl_crit = RewardCriterion()

    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=opt.weight_decay)

    train_patience = 0
    epoch = 0

    loss_meter = meter.AverageValueMeter()

    while True:
        loss_meter.reset()

        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 and opt.optim != 'adadelta':
            frac = int((epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every)
            decay_factor = opt.learing_rate_decay_rate ** frac
            current_lr = opt.learing_rate * decay_factor
            set_lr(optimizer, current_lr)

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_pos)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        for iter, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, lens, gts, video_id) in enumerate(train_loader):
            # print('data.type is ', type(data))
            # print('caps.type is ', type(caps))
            # print('caps_mask.type is ', type(caps_mask))
            # data = data.to(device)
            caps = caps.to(device)
            caps_mask = caps_mask.to(device)
            cap_classes = cap_classes.to(device)
            feats0 = feats0.to(device)
            feats1 = feats1.to(device)
            feat_mask = feat_mask.to(device)
            class_masks = class_masks.to(device)
            # lens = lens.to(device)
            # gts = gts.to(device)
            # video_id = video_id.to(device)
            # print('########caps.device is ', caps.device, '########')
            # print('########caps_mask.device is ', caps_mask.device, '########')
            # print('########feats0.device is ', feats0.device, '########')
            # print('########feats1.device is ', feats1.device, '########')
            # print('########feat_mask.device is ', feat_mask.device, '########')

            cap_classes = torch.cat([cap_classes[:, -1:], cap_classes[:, :-1]], dim=1)
            new_mask = torch.zeros_like(class_masks)
            # print('########cap_classes.device is ', cap_classes.device, '########')
            # print('########new_mask.device is ', new_mask.device, '########')
            cap_classes = cap_classes.to(device)
            new_mask = new_mask.to(device)
            # print('!!!!!!!!!cap_classes.device is ', cap_classes.device, '!!!!!!!!!')
            # print('!!!!!!!!!new_mask.device is ', new_mask.device, '!!!!!!!!!!')
            for i in range(class_masks.size(0)):
                index = np.argwhere(class_masks.cpu().numpy()[i, :] != 0)[0][-1]
                new_mask[i, :index + 1] = 1.0
            optimizer.zero_grad()

            out = model(feats0, feats1, feat_mask, caps, caps_mask, cap_classes, new_mask)
            loss = classify_crit(out, cap_classes, caps_mask, class_masks)

            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.detach()
            # torch.cuda.synchronize()
            # end = time.time()
            loss_meter.add(train_loss.item())

            if (iter + 1) % opt.visualize_every == 0:
                vis.plot('loss', loss_meter.value()[0])

            # print('opt.save_checkpoint_every == ', opt.save_checkpoint_every)
            if (iter + 1) % opt.save_checkpoint_every == 0:
                print('i am saving!!')
                eval_kwargs = {}
                eval_kwargs.update(vars(opt))
                val_loss = eval_extract.eval_and_extract(model, classify_crit, valid_dataset, device, 'validation', eval_kwargs, False)
                # 此处输出的是交叉熵loss, 是一个正数, loss越小说明准确性越高, 为了转化为score我们需要加一个负号, score越大准确性越高
                current_score = -val_loss
                is_best = False
                if best_val_score is None or best_val_score < current_score:
                    best_val_score = current_score
                    is_best = True
                    train_patience = 0
                else:
                    train_patience += 1
                path_of_save_model = os.path.join(opt.checkpoint_path, str(iter) + '_model.pth')
                torch.save(model.state_dict(), path_of_save_model)

                infos['iteration'] = iter
                infos['epoch'] = epoch
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt

                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.model_name + '.pkl'), 'wb') as f:
                    pickle.dump(infos, f)

                if is_best:
                    path_of_save_model = os.path.join(opt.checkpoint_path, opt.model_name + '-bestmodel.pth')
                    torch.save(model.state_dict(), path_of_save_model)
                    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.model_name + '-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

            if train_patience > opt.patience:
                break

        epoch += 1
        if train_patience > opt.patience:
            print('accuracy is not improving any more!!!!!')
            break
        elif epoch > opt.max_epochs:
            print('reached max epoch, now i am breaking out')
            break





if __name__ == '__main__':
    opt = myopts.parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(opt, device)
    pass