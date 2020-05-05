import sys
sys.path.append('../')
import os
import torch
import math

from infersent_model import InferSent
from eval.eval_cap import cosine
import myopts
EPS = 1e-4

if __name__ == '__main__':
    opt = myopts.parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_version = 1
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
    infersent_model.build_vocab_k_words(K=100000)

    store = ['a man is talking about a movie pictures of a movie pictures' ,
             'a person is folding paper',
             'a man is singing',
             'people are dancing and dancing',
             'a man and woman are talking about something',
             'a woman is applying makeup',
             'a person is cooking a dish and adding ingredients into a pot',
             'a man is talking',
             'a man is talking about the weather on the screen',
             'cartoon characters are interacting']
    embeddings = infersent_model.encode(store, bsize=128, tokenize=True)

    for i in range(len(store)):
        temp = infersent_model.encode([store[i]], bsize=128, tokenize=True)[0]
        if math.fabs(1 - cosine(temp, embeddings[i])) > EPS:
            print(cosine(temp, embeddings[i]))