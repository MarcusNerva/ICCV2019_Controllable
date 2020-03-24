import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
sys.path.append('../')
from eval.eval_extract import eval_and_extract
from models.pos_generator import Pos_generator
from models.loss import LanguageModelCriterion, ClassifierCriterion, RewardCriterion
from data.dataset import load_dataset_pos
import myopts

if __name__ == '__main__':
    opt = myopts.parse_opt()
    model = Pos_generator(opt=opt)
    classify_crit = ClassifierCriterion()
    model.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_name + '-bestmodel.pth')))
    train_set, valid_set, test_set = load_dataset_pos(opt=opt)
    # train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    # valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    # test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)

    eval_kwargs = {}
    eval_kwargs.update(vars(opt))
    loss_train = eval_and_extract(model=model, classify_crit=classify_crit, dataset=train_set, dataset_name='train', eval_kwargs=vars(opt), extract_pos=True)
    loss_valid = eval_and_extract(model=model, classify_crit=classify_crit, dataset=valid_set, dataset_name='valid', eval_kwargs=vars(opt), extract_pos=True)
    loss_test = eval_and_extract(model=model, classify_crit=classify_crit, dataset=test_set, dataset_name='test', eval_kwargs=vars(opt), extract_pos=True)
    print('loss_train == ', loss_train)
    print('loss_valid == ', loss_valid)
    print('loss_test == ', loss_test)
