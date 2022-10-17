import json 
import argparse
import numpy as np

_CLASS_NAME = {
    "THUMOS":['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
              'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
              'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
              'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
              'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'],

    "ActivityNet":['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle',
                    'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling',
                    'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves',
                    'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles',
                    'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira',
                    'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood',
                    'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws',
                    'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree',
                    'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing',
                    'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics',
                    'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof',
                    'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo',
                    'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
                    'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop',
                    'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying',
                    'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding',
                    'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks',
                    'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture',
                    'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton',
                    'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums',
                    'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey',
                    'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball',
                    'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin',
                    'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta',
                    'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting',
                    'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing',
                    'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping',
                    'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs',
                    'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining',
                    'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning',
                    'Spread mulch','Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming',
                    'Swinging at the playground', 'Table soccer','Tai chi', 'Tango', 'Tennis serve with ball bouncing',
                    'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars',
                    'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine',
                    'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes',
                    'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents',
                    'Zumba']
}


_DATASET_HYPER_PARAMS = {
    
    "THUMOS":{
        "dropout":0.7,
        "lr":1e-4,
        "weight_decay":5e-5,
        "frames_per_sec":25,
        "segment_frames_num":16,
        "sample_segments_num":750,
        
        "feature_dim":2048,
        "action_cls_num":len(_CLASS_NAME["THUMOS"]),
        "cls_threshold":0.25,
        "test_upgrade_scale":20,
        # "data_dir":"/DATA/W-TAL/THU14/",
        "data_dir":"./data/THUMOS14/",
        "test_gt_file":"./data/THUMOS14/gt.json",
        "tiou_thresholds":np.arange(0.1, 1.00, 0.10),
        "nms_thresh":0.55,
        
        "ins_topk_seg":8,
        "con_topk_seg":3,
        "bak_topk_seg":3,
        
        "loss_lamb_1":2e-3,
        "loss_lamb_2":5e-5,
        "loss_lamb_3":2e-4,
        
    },
    
    "ActivityNet":{
        "dropout":0.7,
        "lr":2e-5,
        "weight_decay":0.001,
        "frames_per_sec":25,
        "segment_frames_num":16,
        "sample_segments_num":75,

        "inp_feat_num":2048,
        "out_feat_num":2048,
        "scale_factor":10.0,

        "action_cls_num":len(_CLASS_NAME["ActivityNet"]),
        "cls_threshold":0.02,
        "test_upgrade_scale":20,
        "data_dir":"/home/jlhuang/code/Data/WSD_Data/ActivityNet1.3_ACM",
        "test_gt_file":"/home/jlhuang/code/Data/WSD_Data/ActivityNet1.3_ACM/gt.json",
        "tiou_thresholds":np.arange(0.50, 1.00, 0.05),
        "act_thresholds":np.arange(0.001, 0.07, 0.001),
        "att_thresholds":np.arange(0.0000, 0.001, 0.0002),
        "nms_thresh":0.75,

        "warmup_epoch":40,

        "n_mu":8,
        'em_iter':2,
        "momentum":0.99,

        "video_num":9032,
        "sample_num":2,

        "o_weight":0.8,
        "m_weight":0.2,

        "lambda_b":0.1,
        "lambda_att":0.1,
        "lambda_spl":1.0,
        "propotion":8.0,
        "temperature":1.0,
        "weight":0.5,
    }} 

def build_args(dataset=None):
    
    parser = argparse.ArgumentParser("This script is used for the weakly-supervised temporal aciton localization task.")
    
    parser.add_argument("--checkpoint", default='', type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gpu", default='1', type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="THUMOS", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    
    parser.add_argument("--without_wandb", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    if dataset is not None:
        args.dataset = dataset
    # Based on the selected dataset, we set dataset specific hyper-params. 
    args.class_name_lst = _CLASS_NAME[args.dataset]
    args.action_cls_num = _DATASET_HYPER_PARAMS[args.dataset]["action_cls_num"]
    
    args.dropout = _DATASET_HYPER_PARAMS[args.dataset]["dropout"]
    args.lr = _DATASET_HYPER_PARAMS[args.dataset]["lr"]
    args.weight_decay = _DATASET_HYPER_PARAMS[args.dataset]["weight_decay"]
    
    args.frames_per_sec = _DATASET_HYPER_PARAMS[args.dataset]["frames_per_sec"]
    args.segment_frames_num = _DATASET_HYPER_PARAMS[args.dataset]["segment_frames_num"]
    args.sample_segments_num = _DATASET_HYPER_PARAMS[args.dataset]["sample_segments_num"]

    args.inp_feat_num =  _DATASET_HYPER_PARAMS[args.dataset]["inp_feat_num"]
    args.out_feat_num =  _DATASET_HYPER_PARAMS[args.dataset]["out_feat_num"]
    args.scale_factor =  _DATASET_HYPER_PARAMS[args.dataset]["scale_factor"]

    args.cls_threshold = _DATASET_HYPER_PARAMS[args.dataset]["cls_threshold"]
    args.tiou_thresholds = _DATASET_HYPER_PARAMS[args.dataset]["tiou_thresholds"]
    args.act_thresholds = _DATASET_HYPER_PARAMS[args.dataset]["act_thresholds"]
    args.att_thresholds = _DATASET_HYPER_PARAMS[args.dataset]["att_thresholds"]
    args.test_gt_file_path = _DATASET_HYPER_PARAMS[args.dataset]["test_gt_file"]
    args.data_dir = _DATASET_HYPER_PARAMS[args.dataset]["data_dir"]

    args.test_upgrade_scale = _DATASET_HYPER_PARAMS[args.dataset]["test_upgrade_scale"]
    args.nms_thresh = _DATASET_HYPER_PARAMS[args.dataset]["nms_thresh"]

    args.warmup_epoch = _DATASET_HYPER_PARAMS[args.dataset]["warmup_epoch"]

    args.n_mu = _DATASET_HYPER_PARAMS[args.dataset]["n_mu"]
    args.em_iter = _DATASET_HYPER_PARAMS[args.dataset]["em_iter"]
    args.momentum = _DATASET_HYPER_PARAMS[args.dataset]["momentum"]

    args.video_num = _DATASET_HYPER_PARAMS[args.dataset]["video_num"]
    args.sample_num = _DATASET_HYPER_PARAMS[args.dataset]["sample_num"]

    args.o_weight = _DATASET_HYPER_PARAMS[args.dataset]["o_weight"]
    args.m_weight = _DATASET_HYPER_PARAMS[args.dataset]["m_weight"]

    args.lambda_b = _DATASET_HYPER_PARAMS[args.dataset]["lambda_b"]
    args.lambda_att = _DATASET_HYPER_PARAMS[args.dataset]["lambda_att"]
    args.lambda_spl = _DATASET_HYPER_PARAMS[args.dataset]["lambda_spl"]
    args.propotion = _DATASET_HYPER_PARAMS[args.dataset]["propotion"]
    args.temperature = _DATASET_HYPER_PARAMS[args.dataset]["temperature"]
    args.weight = _DATASET_HYPER_PARAMS[args.dataset]["weight"]

    return args