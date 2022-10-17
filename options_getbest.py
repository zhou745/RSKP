import argparse

parser = argparse.ArgumentParser(description='WSTAL')
# seed 2 1986 23068
# basic setting
parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--run-type', type=int, default=0,
                    help='train (0) or evaluate (1)')
parser.add_argument('--model-id', type=str, default="rskp_dist_linprog_2", help='model id for saving model')
# loading model
parser.add_argument('--pretrained', action='store_true',default=True, help='is pretrained model')
parser.add_argument('--use_new_predictor',type=bool, default=False, help='if the fusion predioctor is used')
parser.add_argument('--load_epoch', type=int, default=200)
parser.add_argument('--pretrained_epoch', type=int, default=200)
parser.add_argument('--load_ckpt_path', type=str, default='./ckpt/Thumos14reduced/rskp_baseline_pretraining_2', help='epoch of loaded model')


# storing parameters
parser.add_argument('--save-interval', type=int, default=1, help='interval for storing model')

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
parser.add_argument('--dropout', default=0.5, help='dropout value (default: 0.5)')
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
parser.add_argument('--max-epoch', type=int, default=300, help='maximum iteration to train (default: 50000)')

parser.add_argument('--mu-num', type=int, default=8, help='number of Gaussians')
parser.add_argument('--mu-queue-len', type=int, default=5, help='number of slots of each class of memory bank')
parser.add_argument('--em-iter', type=int, default=2, help='number of EM iteration')



parser.add_argument('--warmup-epoch', default=100, help='epoch starting to use the inter-video branch')

# testing paramaters
parser.add_argument('--class-threshold', type=float, default=0.16, help='class threshold for rejection')
parser.add_argument('--start-threshold', type=float, default=0.001, help='start threshold for action localization')
parser.add_argument('--end-threshold', type=float, default=0.04, help='end threshold for action localization')
parser.add_argument('--threshold-interval', type=float, default=0.002, help='threshold interval for action localization')
