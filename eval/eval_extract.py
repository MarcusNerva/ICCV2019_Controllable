import torch
import torch.nn as nn
import numpy as np
import sys
import os
import h5py
sys.path.append('../')
from data.dataset import *
from torch.utils.data import DataLoader
from models.pos_generator import *

def eval_and_extract(model, classify_crit, dataset, device, dataset_name='train' , eval_kwargs={}, extract_pos=False):
    lang_eval = eval_kwargs.get('language_eval', 0)
    beam_size = eval_kwargs.get('beam_size', 1)
    weight_class = eval_kwargs.get('weight_class', 0.0)

    model.eval()

    loss_sum = 0
    loss_evals = 1e-8

    if extract_pos:
        print(eval_kwargs['data_path'])
        path = os.path.join(eval_kwargs['data_path'], '/pos_features/' + dataset_name + '.hdf5')
        print('path is ', path)
        # if not os.path.exists(path):
        #     open(path)
        writer = h5py.File(path, 'w')
    dataset_loader = DataLoader(dataset, batch_size=eval_kwargs.get('batch_size', 64), shuffle=True, collate_fn=collate_fn_pos)
    for iter, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, lens, gts, video_id) in enumerate(dataset_loader):
        caps = caps.to(device)
        caps_mask = caps_mask.to(device)
        cap_classes = cap_classes.to(device)
        class_masks = class_masks.to(device)
        feats0 = feats0.to(device)
        feats1 = feats1.to(device)
        feat_mask = feat_mask.to(device)
        # print('######here comes cap_classes#######')
        # print(cap_classes)

        cap_classes = torch.cat([cap_classes[:, -1:], cap_classes[:, :-1]], dim=-1)
        # 先在这里把最后一个放在第一位(<EOS>和<BOS>貌似没有区分), 这样out就可以生成和cap_classes未变化之前同样的结构,
        # 之后在classify_crit中把cap_classes再变回来, 这样就可以实现out和cap_classes同结构比较了
        new_mask = torch.zeros_like(class_masks)
        for i in range(class_masks.size(0)):
            index = np.argwhere(class_masks.cpu().numpy()[i, :] != 0)[0][-1]
            new_mask[i, :index + 1] = 1.0
        out = model(feats0, feats1, feat_mask, caps, caps_mask, cap_classes, class_masks)
        # print('input(out).shape == ', out.shape)
        # print('cap_classes.shape == ', cap_classes.shape)
        # print('caps_mask.shape == ', caps_mask.shape)
        loss = classify_crit(out, cap_classes, caps_mask, class_masks).detach()
        loss_sum += loss
        loss_evals += 1

        seq, seqLogprobs, collect_state, collect_mask = model.sample(feats0, feats1, feat_mask, eval_kwargs)
        seq = seq.cpu()
        seqLogprobs = seqLogprobs.cpu()

        collect_state = collect_state.data.cpu().numpy()
        collect_mask = collect_mask.data.cpu().numpy()
        collect_seq = seq.numpy()

        if extract_pos:
            for i, vid in enumerate(video_id):
                try:
                    writer.create_group(vid)
                except ValueError:
                    continue
                writer[vid]['states'] = collect_state[i]
                writer[vid]['masks'] = collect_mask[i:i + 1, :]
                writer[vid]['tokens'] = collect_seq[i:i + 1, :]

        if extract_pos:
            writer.close()

        model.train()
        return loss_sum / loss_evals

