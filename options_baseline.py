import argparse

parser = argparse.ArgumentParser(description='WSTAL')
# seed 2 1986 23068
# basic setting
parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--run-type', type=int, default=0,
                    help='train (0) or evaluate (1)')
parser.add_argument('--model-id', type=str, default="rskp_baseline_tmp", help='model id for saving model')
# loading model
parser.add_argument('--pretrained', default= False,action='store_true', help='is pretrained model')
parser.add_argument('--load-epoch', type=int, default=None, help='epoch of loaded model')
parser.add_argument('--use_new_predictor', type=bool, default=False, help='whether to use the new prediction module')
# storing parameters
parser.add_argument('--save-interval', type=int, default=5, help='interval for storing model')

# dataset patameters
parser.add_argument('--dataset-root', default='./data/', help='dataset root path')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on')
parser.add_argument('--video-num', default=200, help='video number')

# model settings
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used (default: I3D)')
parser.add_argument('--inp-feat-num', type=int, default=2048, help='size of input feature (default: 2048)')
parser.add_argument('--out-feat-num', type=int, default=2048, help='size of output feature (default: 2048)')
parser.add_argument('--class-num', type=int, default=20, help='number of classes (default: 20)')
parser.add_argument('--scale-factor', type=float, default=20.0, help='temperature factors')

parser.add_argument('--T', type=float, default=0.2, help='number of head')
parser.add_argument('--w', type=float, default=0.5, help='number of head')

# training paramaters
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--lr-decay', type=float, default=0.8, help='learning rate decay(default: 0.0001)')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight deacy (default: 0.001)')
parser.add_argument('--dropout', default=0.6, help='dropout value (default: 0.5)')
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
parser.add_argument('--max-epoch', type=int, default=200, help='maximum iteration to train (default: 50000)')

parser.add_argument('--mu-num', type=int, default=8, help='number of Gaussians')
parser.add_argument('--mu-queue-len', type=int, default=5, help='number of slots of each class of memory bank')
parser.add_argument('--em-iter', type=int, default=2, help='number of EM iteration')



parser.add_argument('--warmup-epoch', default=100, help='epoch starting to use the inter-video branch')

# testing paramaters
parser.add_argument('--class-threshold', type=float, default=0.16, help='class threshold for rejection')
parser.add_argument('--start-threshold', type=float, default=0.001, help='start threshold for action localization')
parser.add_argument('--end-threshold', type=float, default=0.04, help='end threshold for action localization')
parser.add_argument('--threshold-interval', type=float, default=0.002, help='threshold interval for action localization')

# Learning Rate Decay
parser.add_argument('--decay-type', type=int, default=1, help='weight decay type (0 for None, 1 for step decay, 2 for cosine decay)')
parser.add_argument('--changeLR_list', type=int, default=[80,1000], help='change lr step')
parser.add_argument('--use_mem', type=int, default=1, help='0 not use 1 use')
parser.add_argument('--use_foreloss', type=int, default=1, help='0 not use 1 use')
parser.add_argument('--use_backloss', type=int, default=1, help='0 not use 1 use')
parser.add_argument('--use_attloss', type=int, default=1, help='0 not use 1 use')

parser.add_argument('--frm_coef', type=float, default=0.85, help='mix up pred and mu')
parser.add_argument('--fore_loss_weight', type=float, default=1, help='mix up pred and mu')
parser.add_argument('--spl_loss_weight', default=1., help='weight of pseudo label supervision loss')
parser.add_argument('--back_loss_weight', default=0.2, help='weight of pseudo label supervision loss')
parser.add_argument('--att_loss_weight', default=0.1, help='weight of attention normalization loss')
