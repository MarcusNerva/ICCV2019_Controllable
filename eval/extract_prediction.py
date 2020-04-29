import sys
sys.path.append('../')
from torch.utils.data import DataLoader
from models.describer_generator import Caption_generator
from eval.eval_cap import decode_idx

def generate_prediction(model:Caption_generator, dataset_loader:DataLoader, kwargs:dict = {}) -> dict:
    data_path = kwargs.get('data_path', None)
    assert data_path is not None, 'The data_path is not exist!'

    model.eval()
    ret = {}
    id_word = get_itow(data_path)
    for i, (data, caps, caps_mask, cap_classes, class_masks, feats0, feats1, feat_mask, pos_feat, lens, gts, video_id) in enumerate(dataset_loader):
        feats0 = feats0.to(device)
        feats1 = feats1.to(device)
        feat_mask = feat_mask.to(device)
        pos_feat = pos_feat.to(device)
        caps = caps.to(device)
        caps_mask = caps_mask.to(device)
        cap_classes = cap_classes.to(device)
        class_masks = class_masks.to(device)

        seq, seq_probabilities = model.sample(feats0, feats1, feat_mask, pos_feat, kwargs)
        seqs = seq.cpu().numpy()

        for t in range(seqs.shape[0]):
            prediction = decode_idx(seqs[t], id_word)
            vid_t = video_id[t].decode()
            ret[vid_t] = prediction

    return ret


if __name__ == '__main__':
    import myopts
    import pickle
    import torch
    import os
    import numpy as np
    from data.dataset import load_dataset_cap_for_prediction, collate_fn_cap, get_itow, get_caps, get_nwords, load_pkl

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = myopts.parse_opt()
    train_dataset, valid_dataset, test_dataset = load_dataset_cap_for_prediction(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_cap)

    model = Caption_generator(opt)
    assert opt.start_from is not None, 'opt.start_from is None!'
    state_dict_path = os.path.join(opt.start_from, 'cap_2e-4_0.5_4_xavier.pth')
    model.load_state_dict(torch.load(state_dict_path))
    model = model.to(device)
    kwargs = vars(opt)

    train_prediction = generate_prediction(model=model, dataset_loader=train_dataloader, kwargs=kwargs)
    valid_prediction = generate_prediction(model=model, dataset_loader=valid_dataloader, kwargs=kwargs)
    test_prediction = generate_prediction(model=model, dataset_loader=test_dataloader, kwargs=kwargs)

    total_dict = {}
    total_dict.update(train_prediction)
    total_dict.update(valid_prediction)
    total_dict.update(test_prediction)

    prediction_path = os.path.join(opt.data_path, 'prediction.pkl')
    with open(prediction_path, 'wb') as f:
        pickle.dump(total_dict, f)
