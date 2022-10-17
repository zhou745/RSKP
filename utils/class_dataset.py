import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset

class_name_lst = ['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle',
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

def uniform_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len <= sample_len and input_len > 1:
        sample_idxs = np.arange(input_len)
    else:
        if input_len == 1:
            sample_len = 2
        sample_scale = input_len / sample_len
        sample_idxs = np.arange(sample_len) * sample_scale
        sample_idxs = np.floor(sample_idxs)

    return input_feature[sample_idxs.astype(np.int), :]


def random_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len < sample_len:
        sample_idxs = np.random.choice(input_len, sample_len, replace=True)
        sample_idxs = np.sort(sample_idxs)
    elif input_len > sample_len:
        sample_idxs = np.arange(sample_len) * input_len / sample_len
        for i in range(sample_len - 1):
            sample_idxs[i] = np.random.choice(range(np.int(sample_idxs[i]), np.int(sample_idxs[i + 1] + 1)))
        sample_idxs[-1] = np.random.choice(np.arange(sample_idxs[-2], input_len))
    else:
        sample_idxs = np.arange(input_len)

    return input_feature[sample_idxs.astype(np.int), :]


def consecutive_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len >= sample_len:
        sample_idx = np.random.choice((input_len - sample_len))
        return input_feature[sample_idx:(sample_idx + sample_len), :]

    elif input_len < sample_len:
        empty_features = np.zeros((sample_len - input_len, input_feature.shape[1]))
        return np.concatenate((input_feature, empty_features), axis=0)


class ACMDataset(Dataset):

    def __init__(self, args, phase="train", sample="random"):

        self.phase = phase
        self.sample = sample
        self.data_dir = os.path.join(args.dataset_root,args.dataset_name)
        self.sample_segments_num = args.sample_segments_num

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]

        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]

        self.class_name_lst = class_name_lst
        self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}

        self.action_class_num = len(class_name_lst)

        self.get_label()

    def get_label(self):

        self.label_dict = {}
        for item_name in self.data_list:

            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0

            self.label_dict[item_name] = item_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_duration = self.gt_dict[vid_name]["duration"]
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name + ".npy"))

        vid_len = con_vid_feature.shape[0]

        if self.sample == "random":
            con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
        else:
            con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)

        con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32))

        vid_label_t = torch.as_tensor(vid_label.astype(np.float32))

        if self.phase == "train":
            return con_vid_spd_feature, vid_label_t
        else:
            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration


def build_dataset(args, phase="train", sample="random"):
    return ACMDataset(args, phase, sample)