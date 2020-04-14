import torch
import numpy as np
import sys
import os
sys.path.append('../')
from models.describer_generator import Caption_generator
from eval.eval_cap import eval
from data.dataset import load_dataset_cap, get_itow
from models.loss import LanguageModelCriterion, ClassifierCriterion, RewardCriterion
import myopts

if __name__ == '__main__':
    opt = myopts.parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset, test_dataset = load_dataset_cap(opt)
    model = Caption_generator(opt)
    assert opt.start_from is not None, 'opt.start_from should not be Nonetype!'
    state_dict_path = os.path.join(opt.start_from, opt.model_name + '-bestmodel.pth')
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    crit = LanguageModelCriterion()
    classify_crit = ClassifierCriterion()
    if not opt.eval_semantics:
        avg_loss, language_state = eval(model, crit, classify_crit, test_dataset, vars(opt))
        print(language_state)
    else:
        textual_score, language_state = eval(model, crit, classify_crit, test_dataset, vars(opt))
        print('textual_score is ', textual_score)
        print(language_state)