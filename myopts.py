import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_path', type=str,
    #                     default='/Users/bismarck/PycharmProjects/ICCV2019_Controllable/data',
    #                     help='the path of the data')
    parser.add_argument('--data_path', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/ICCV2019_Controllable/data',
                        help='the path of the data')
    # parser.add_argument('--start_from', type=str,
    #                     default='/Users/bismarck/PycharmProjects/ICCV2019_Controllable/checkpoints',
    #                     help='continue training from saved model at this path')
    parser.add_argument('--start_from', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/ICCV2019_Controllable/checkpoints',
                        help='continue training from saved model at this path')
    # parser.add_argument('--checkpoint_path', type=str,
    #                     default='/Users/bismarck/PycharmProjects/ICCV2019_Controllable/checkpoints',
    #                     help='the path of saving a trained model and it\'s information')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/ICCV2019_Controllable/checkpoints',
                        help='the path of saving a trained model and it\'s information')
    parser.add_argument('--feat_K', type=int, default=30,
                        help=' the number of feats(frames) take out from a video')

    # Model settings
    parser.add_argument('--vocab_size', type=int, default=20000, help='number of all words')
    parser.add_argument('--category_size', type=int, default=14, help='number of all category')
    parser.add_argument('--rnn_size', type=int, default=512, help='size of the LSTM\'s hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM')
    parser.add_argument('--seq_length', type=int, default=28, help='the length of sentences from ground truth')
    parser.add_argument('--word_embed_size', type=int, default=468, help='the encoded size of word embedding')
    parser.add_argument('--att_size', type=int, default=1536, help='the size of the attention mechinism')
    parser.add_argument('--feat0_size', type=int, default=1536, help='the size of the rgb feature size')
    parser.add_argument('--feat1_size', type=int, default=1024, help='the size of the opfl feature size')
    parser.add_argument('--activity_fn', type=str, default='ReLU', help='ReLU, Tanh, Sigmoid...')
    parser.add_argument('--weight_class', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0, help='the random seed')

    # General settings
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--drop_probability', type=float, default=0.5, help='strength of dropout')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size')
    parser.add_argument('--save_checkpoint_every', type=int, default=50, help='save a model every x iteration')
    parser.add_argument('--reward_type', type=str, default='CIDEr', help='use BLEU/METEOR/ROUGE/CIDEr as reward')
    parser.add_argument('--patience', type=int, default=30, help='the early stop threshold which designed for solving the score stopped raising')
    parser.add_argument('--model_name', type=str, default='pos', help='name of the model under using')
    parser.add_argument('--load_best_score', type=int, default=0, help='if you want to load previous best score, please input 1. Otherwise input 0.')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, help='after how many iteration begin learning rate decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4, help='for every x iteration learning rate have to decay')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--optim', type=str, default='adam', help='the optimizer\'s type: adam or adadelta')
    parser.add_argument('--visualize_every', type=int, default=3, help='show us loss every x iteration')


    args = parser.parse_args()

    return args