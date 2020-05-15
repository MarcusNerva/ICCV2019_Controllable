import torch
import numpy as np
import sys
import os
sys.path.append('../')
from models.describer_generator import Caption_generator
from eval.eval_cap import eval
from data.dataset import load_dataset_cap, get_itow
from models.loss import LanguageModelCriterion, ClassifierCriterion, RewardCriterion
from infersent_model import InferSent
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

    model_version = 2
    MODEL_PATH = opt.infersent_model_path
    assert MODEL_PATH is not None, '--infersent_model_path is None!'
    MODEL_PATH = os.path.join(MODEL_PATH, 'infersent%s.pkl' % model_version)
    params_model = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': model_version
    }
    infersent_model = InferSent(params_model)
    infersent_model.load_state_dict(torch.load(MODEL_PATH))
    infersent_model = infersent_model.to(device)
    W2V_PATH = opt.w2v_path
    assert W2V_PATH is not None, '--w2v_path is None!'
    infersent_model.set_w2v_path(W2V_PATH)
    # sentences_path = os.path.join(opt.data_path, 'sentences.pkl')
    # sentences = load_pkl(sentences_path)
    # infersent_model.build_vocab(sentences, tokenize=True)
    infersent_model.build_vocab_k_words(K=100000)

    if not opt.eval_semantics:
        avg_loss, language_state = eval(infersent_model, model, crit, classify_crit, test_dataset, eval_kwargs=vars(opt))
        print(language_state)
    else:
        semantics_score, language_state = eval(infersent_model, model, crit, classify_crit, test_dataset, eval_kwargs=vars(opt))
        print('semantics_score is ', semantics_score)
        print(language_state)