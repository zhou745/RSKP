import os 
import json 
import time
import pickle
from tqdm import tqdm 

import wandb 
import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config.model_config import build_args 
from dataset.dataset_class import build_dataset
from model.Model import WSNet, random_walk
from model.Memory import Memory
from utils.net_utils import set_random_seed, WSLoss
from utils.net_evaluation import ANETDetection, upgrade_resolution, get_proposal_oic, smooth, nms, result2json


"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TRAIN FUNCTION                                                   #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""

def train(args, epoch_idx, model, memory, dataloader, criterion, optimizer):
    model.train()
    print("-------------------------------------------------------------------------------")
    device = args.device
    
    # train_process
    train_num_correct = 0
    train_num_total = 0
    
    loss_stack = []
    fg_loss_stack = []
    bg_loss_stack = []    
    att_loss_stack = []
    spl_loss_stack = []

    for idx, input_feature, vid_label_t, vid_name in tqdm(dataloader):

        vid_label = vid_label_t.to(device)
        input_feature = input_feature.to(device)
        vid_label_np = vid_label_t.cpu().numpy()

        o_out, m_out, em_out = model(input_feature)

        if epoch_idx >= args.warmup_epoch:
            select_vid_mu = memory._return_queue(idx, vid_label_np).detach()
            reallocated_x = random_walk(em_out[1], select_vid_mu, args.weight)
            r_out = model.PredictionModule(reallocated_x)
            loss, loss_dict = criterion(o_out, m_out, r_out, vid_label)
            memory._update_queue(em_out[0].squeeze(0), idx)
        else:
            loss, loss_dict = criterion(o_out, m_out, None, vid_label)  

        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            act_cls = args.o_weight * o_out[0] + args.m_weight * m_out[0]
            fg_score = act_cls[:, :args.action_cls_num]
            label_np = vid_label_t.cpu().numpy()
            score_np = fg_score.cpu().numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[score_np >= args.cls_threshold] = 1
            pred_np[score_np < args.cls_threshold] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)
            
            train_num_correct += np.sum((correct_pred == args.action_cls_num))
            train_num_total += correct_pred.shape[0]
            
            loss_stack.append(loss.cpu().item())
            fg_loss_stack.append(loss_dict["fg_loss"])
            bg_loss_stack.append(loss_dict["bg_loss"])
            att_loss_stack.append(loss_dict["att_loss"])
            spl_loss_stack.append(loss_dict["spl_loss"])

    train_acc = train_num_correct / train_num_total

    train_log_dict = {}
    train_log_dict["train_fg_loss"] = np.mean(fg_loss_stack)
    train_log_dict["train_bg_loss"] = np.mean(bg_loss_stack)
    train_log_dict["train_att_loss"] = np.mean(att_loss_stack)
    train_log_dict["train_spl_loss"] = np.mean(spl_loss_stack)
    train_log_dict["train_loss"] = np.mean(loss_stack)
    train_log_dict["train_acc"] = train_acc

    print("")
    print("train_fg_loss:{:.3f}  train_bg_loss:{:.3f}".format(np.mean(fg_loss_stack), np.mean(bg_loss_stack)))
    print("train_att_loss:{:.3f}  train_spl_loss:{:.3f}".format(np.mean(att_loss_stack), np.mean(spl_loss_stack)))
    print("train_loss:{:.3f}  train acc:{:.3f}".format(np.mean(loss_stack), train_acc))
    print("-------------------------------------------------------------------------------")

    return train_log_dict

"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TEST FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""

def test(args, model, dataloader, criterion, phase="test"):
    
    model.eval()
    print("-------------------------------------------------------------------------------")
    device = args.device
    save_dir = args.save_dir
    
    test_num_correct = 0
    test_num_total = 0
    
    loss_stack = []
    fg_loss_stack = []
    bg_loss_stack = []
    att_loss_stack = []
    spl_loss_stack = []

    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}
    
    test_tmp_data_log_dict = {}
    
    for vid_name, input_feature, vid_label_t, vid_len, vid_duration in tqdm(dataloader):
        
        input_feature = input_feature.to(device)
        vid_label_t = vid_label_t.to(device)
        vid_len = vid_len[0].cpu().numpy()
        t_factor = (args.segment_frames_num * vid_len) / (args.frames_per_sec * args.test_upgrade_scale * input_feature.shape[1])
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        o_out, m_out, em_out = model(input_feature)
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        loss, loss_dict = criterion(o_out, m_out, None, vid_label_t) 

        loss_stack.append(loss.cpu().item())
        fg_loss_stack.append(loss_dict["fg_loss"])
        bg_loss_stack.append(loss_dict["bg_loss"])
        att_loss_stack.append(loss_dict["att_loss"])
        spl_loss_stack.append(loss_dict["spl_loss"])
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        o_cas = F.softmax(o_out[3], -1)
        m_cas = F.softmax(m_out[3], -1)
        temp_cas = args.o_weight * o_cas + args.m_weight * m_cas
        temp_att = o_out[2]
        test_tmp_data_log_dict[vid_name[0]] = {}
        test_tmp_data_log_dict[vid_name[0]]["vid_len"] = vid_len
        test_tmp_data_log_dict[vid_name[0]]["temp_o_cls_score_np"] = o_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_m_cls_score_np"] = m_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cas.cpu().numpy()
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        act_cls = args.o_weight * o_out[0] + args.m_weight * m_out[0]
        fg_score = act_cls[:, :args.action_cls_num]
        label_np = vid_label_t.cpu().numpy()
        score_np = fg_score.cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0
        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_score_np = temp_att.unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_score_np = np.reshape(temp_att_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0])
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=np.int)

        temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
        temp_att_score_np = temp_att_score_np[:, cls_prediction]

        test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cls_score_np
        int_temp_cls_score_np = upgrade_resolution(temp_cls_score_np, args.test_upgrade_scale)
        int_temp_att_score_np = upgrade_resolution(temp_att_score_np, args.test_upgrade_scale)

        # int_temp_cls_score_np = (int_temp_cls_score_np - np.min(int_temp_cls_score_np, axis=0, keepdims=True)) / \
        # (np.max(int_temp_cls_score_np, axis=0, keepdims=True) - np.min(int_temp_cls_score_np, axis=0, keepdims=True) + 1e-6)
        int_temp_cls_score_np = smooth(int_temp_cls_score_np)
        # int_temp_att_score_np = smooth(int_temp_att_score_np)      
        # int_temp_cls_score_np = int_temp_cls_score_np / (np.max(int_temp_cls_score_np, axis=0, keepdims=True) + 1e-6)
        # int_temp_att_score_np = int_temp_att_score_np / (np.max(int_temp_att_score_np, axis=0, keepdims=True) + 1e-6)

        cas_act_thresh = args.act_thresholds
        cas_att_thresh = args.att_thresholds

        proposal_dict = {}
        # CAS based proposal generation
        # cas_act_thresh = []
        for act_thresh in cas_act_thresh:

            tmp_int_cas = int_temp_cls_score_np.copy()
            zero_location = np.where(tmp_int_cas < act_thresh)
            tmp_int_cas[zero_location] = 0
            
            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (0.4 * tmp_int_cas + 0.6 * int_temp_att_score_np), cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.10)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]

        for att_thresh in cas_att_thresh:

            tmp_int_att = int_temp_att_score_np.copy()
            zero_location = np.where(tmp_int_att < att_thresh)
            tmp_int_att[zero_location] = 0

            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                tmp_seg_list.append(pos)
            
            props_list = get_proposal_oic(tmp_seg_list, (0.3 * int_temp_cls_score_np + 0.7 * tmp_int_att), cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.250)
            
            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]
                
                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []
                
                proposal_dict[class_id] += props_list[i]

        # NMS 
        final_proposals = []
        
        for class_id in proposal_dict.keys():
            final_proposals.append(nms(proposal_dict[class_id], args.nms_thresh))
                
        test_final_result['results'][vid_name[0]] = result2json(final_proposals, args.class_name_lst)
        
    test_acc = test_num_correct / test_num_total
    
    if args.test:
        # Final Test
        test_final_json_path = os.path.join(save_dir, "final_test_{}_result.json".format(args.dataset))
    else:
        # Train Evalutaion
        test_final_json_path = os.path.join(save_dir, "{}_lateset_result.json".format(args.dataset))

    with open(test_final_json_path, 'w') as f:
        json.dump(test_final_result, f)

    anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                    prediction_file=test_final_json_path,
                    tiou_thresholds=args.tiou_thresholds,
                    subset="val")
    
    test_mAP = anet_detection.evaluate()
    
    print("")
    print("test_fg_loss:{:.3f}  test_bg_loss:{:.3f}".format(np.mean(fg_loss_stack), np.mean(bg_loss_stack)))
    print("test_att_loss:{:.3f}  test_spl_loss:{:.3f}".format(np.mean(att_loss_stack), np.mean(spl_loss_stack)))
    print("test_loss:{:.3f}  test acc:{:.3f}".format(np.mean(loss_stack), test_acc))
    print("-------------------------------------------------------------------------------")

    test_log_dict = {}
    test_log_dict["test_fg_loss"] = np.mean(fg_loss_stack)
    test_log_dict["test_bg_loss"] = np.mean(spl_loss_stack)
    test_log_dict["test_att_loss"] = np.mean(att_loss_stack)
    test_log_dict["test_spl_loss"] = np.mean(spl_loss_stack)
    test_log_dict["test_loss"] = np.mean(loss_stack)
    test_log_dict["test_acc"] = test_acc
    test_log_dict["test_mAP"] = test_mAP
        
    return test_log_dict, test_tmp_data_log_dict


def generate_mu(args, model, dataloader):
    model.eval()
    print("-------------------------------------------------------------------------------")
    device = args.device

    mu_queue = []
    label_queue = []

    for idx, input_feature, vid_label, vid_name in tqdm(dataloader):
        input_feature = input_feature.to(device)

        with torch.no_grad():
            o_out, m_out, em_out = model(input_feature)
            mu = em_out[0]

        mu_queue.append(mu)
        label_queue.append(vid_label.cpu().numpy())

    return torch.cat(mu_queue, 0), np.concatenate(label_queue, 0)

"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              MAIN FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""   
def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_acmnet", "checkpoints_acmnet_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}"\
                                        .format(local_time[0], local_time[1], local_time[2],\
                                                local_time[3], local_time[4]))
    else:
        save_dir = os.path.dirname(args.checkpoint)

    args.save_dir = save_dir
    args.device = device
        
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    model = WSNet(args)
    memory = Memory(args)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    memory = memory.to(device)

    if not args.test:
        if not args.without_wandb:
            wandb.init(name='traing_log_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'\
                            .format(local_time[0], local_time[1], local_time[2],
                                    local_time[3], local_time[4]),
                    config=args,
                    project="WSNet_{}".format(args.dataset),
                    sync_tensorboard=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        train_dataset = build_dataset(args, phase="train", sample="random")
        train_dataset_tmp = build_dataset(args, phase="train", sample="all")        
        test_dataset = build_dataset(args, phase="test", sample="uniform") 

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                      num_workers=args.num_workers, drop_last=False)

        train_dataloader_tmp = DataLoader(train_dataset_tmp, batch_size=1, shuffle=False, 
                                          num_workers=args.num_workers, drop_last=False)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, drop_last=False)

        criterion = WSLoss(lambda_b=args.lambda_b, lambda_att=args.lambda_att, lambda_spl=args.lambda_spl, \
                           propotion=args.propotion, temperature=args.temperature, weight=args.weight)

        best_test_mAP = 0
        
        for epoch_idx in tqdm(range(args.start_epoch, args.epochs)):
        
            train_log_dict = train(args, epoch_idx, model, memory, train_dataloader, criterion, optimizer)
            if epoch_idx % 2 == 0:
                with torch.no_grad():
                    test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
                    test_mAP = test_log_dict["test_mAP"]
                    
                if test_mAP > best_test_mAP:
                    best_test_mAP = test_mAP
                    checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
                    torch.save({
                        'epoch':epoch_idx,
                        'model_state_dict':model.state_dict()
                        }, os.path.join(save_dir, checkpoint_file))
                                        
                    with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                        pickle.dump(test_tmp_data_log_dict, f)

                checkpoint_file = "{}_latest_checkpoint.pth".format(args.dataset)
                torch.save({
                    'epoch':epoch_idx,
                    'model_state_dict':model.state_dict()
                    }, os.path.join(save_dir, checkpoint_file))
                
                print("Current test_mAP:{:.4f}, Current Best test_mAP:{:.4f} Current Epoch:{}/{}".format(test_mAP, best_test_mAP, epoch_idx, args.epochs))
                print("-------------------------------------------------------------------------------")

            if (epoch_idx + 1) == args.warmup_epoch:
                mu_queue, label_queue = generate_mu(args, model, train_dataloader_tmp)
                memory._init_queue(mu_queue, label_queue)
                # args.lambda_spl = 0.5

            if not args.without_wandb:
                wandb.log(train_log_dict)
                wandb.log(test_log_dict)

    else:
        test_dataset = build_dataset(args, phase="test", sample="uniform")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, drop_last=False)
        criterion = WSLoss()
        
        with torch.no_grad():
            test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
            test_mAP = test_log_dict["test_mAP"]
            
            with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                pickle.dump(test_tmp_data_log_dict, f)
                

if __name__ == "__main__":
    
    torch.manual_seed(2)
    np.random.seed(2)
    # set_random_seed()
    args = build_args(dataset="ActivityNet")
    print(args)
    main(args)