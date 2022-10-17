from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from evaluation.eval import ft_eval, ss_eval
# from model.memory import Memory
from model.Memory_anet import Memory
from model.main_baseline import WSTAL, random_walk
from model.losses import NormalizedCrossEntropy, AttLoss, CategoryCrossEntropy
from utils.class_dataset import build_dataset
# from tensorboard_logger import Logger
from torch.utils.tensorboard import SummaryWriter
from utils.net_evaluation import ANETDetection, upgrade_resolution, get_proposal_oic, nms, result2json, ACMLoss,smooth
from utils.class_dataset import class_name_lst
import json
from utils.utils2 import write_results_to_file

class WSLoss(nn.Module):

    def __init__(self, lambda_b=0.2, lambda_att=0.1, lambda_spl=1.0, propotion=8.0, temperature=0.2, weight=0.5):
        super(WSLoss, self).__init__()
        self.lambda_b = lambda_b
        self.lambda_att = lambda_att
        self.lambda_spl = lambda_spl
        self.propotion = propotion
        self.temperature = temperature
        self.weight = weight

    def NormalizedCrossEntropy(self, pred, labels):
        new_labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-6)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred + 1e-6), dim=1), dim=0)
        return loss

    def CategoryCrossEntropy(self, pred, soft_label):
        soft_label = F.softmax(soft_label / self.temperature, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)

        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / self.temperature, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss

    def AttLoss(self, att):
        t = att.size(1)
        max_att_values, _ = torch.topk(att, max(int(t // self.propotion), 1), -1)
        mean_max_att = max_att_values.mean(1)

        min_att_values, _ = torch.topk(-att, max(int(t // self.propotion), 1), -1)
        mean_min_att = -min_att_values.mean(1)

        loss = (mean_min_att - mean_max_att).mean(0)

        return loss

    def forward(self, o_out, m_out, r_out, vid_label):

        device = o_out[0].device
        batch_size = vid_label.shape[0]

        fg_labels = torch.cat([vid_label, torch.zeros(batch_size, 1).to(device)], -1)
        bg_labels = torch.cat([vid_label, torch.ones(batch_size, 1).to(device)], -1)

        # classification loss
        if r_out is None:
            fg_loss = self.NormalizedCrossEntropy(o_out[0], fg_labels) \
                      + self.NormalizedCrossEntropy(m_out[0], fg_labels)
            bg_loss = self.NormalizedCrossEntropy(o_out[1], bg_labels) \
                      + self.NormalizedCrossEntropy(m_out[1], bg_labels)
        else:
            fg_loss = self.NormalizedCrossEntropy(o_out[0], fg_labels) \
                      + self.NormalizedCrossEntropy(m_out[0], fg_labels) \
                      + self.NormalizedCrossEntropy(r_out[0], fg_labels) * 0.5
            bg_loss = self.NormalizedCrossEntropy(o_out[1], bg_labels) \
                      + self.NormalizedCrossEntropy(m_out[1], bg_labels) \
                      + self.NormalizedCrossEntropy(r_out[1], bg_labels) * 0.5

        cls_loss = fg_loss + bg_loss * self.lambda_b

        # attention loss
        att_loss = self.AttLoss(o_out[2])

        # cross branch supervision
        if r_out is None:
            spl_loss = self.CategoryCrossEntropy(o_out[3], m_out[3])
        else:
            spl_loss = self.CategoryCrossEntropy(o_out[3], 0.2 * r_out[3] + 0.8 * m_out[3])

        # total loss
        loss = cls_loss + att_loss * self.lambda_att + spl_loss * self.lambda_spl

        loss_dict = {}
        loss_dict["fg_loss"] = fg_loss.cpu().item()
        loss_dict["bg_loss"] = bg_loss.cpu().item()
        loss_dict["att_loss"] = att_loss.cpu().item()
        loss_dict["spl_loss"] = spl_loss.cpu().item()

        return loss, loss_dict


def generate_mu(args, model, dataloader):
    model.eval()
    print("-------------------------------------------------------------------------------")
    device = torch.device("cuda:0")

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

class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # create logger
        log_dir = './logs/' + self.args.dataset_name + '/' + str(self.args.model_id)
        # self.logger = Logger(log_dir)
        self.logger = SummaryWriter(log_dir)
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')



        # dataloader
        if self.args.dataset_name in ['Anet']:
            if self.args.run_type == 0:
                self.train_dataset = build_dataset(self.args, phase="train", sample="random")
                self.train_dataset_tmp = build_dataset(self.args, phase="train", sample="all")
                self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                                     batch_size=self.args.batch_size,
                                                                     shuffle=True,
                                                                     num_workers=16,
                                                                     drop_last=False)
                self.train_data_loader_tmp = torch.utils.data.DataLoader(self.train_dataset_tmp, batch_size=1,
                                                                         shuffle=False, drop_last=False, num_workers=0)
                self.test_data_loader = torch.utils.data.DataLoader(build_dataset(self.args, phase="test", sample="uniform"),
                                                                    batch_size=1,shuffle=False, drop_last=False, num_workers=0)
        else:
            raise ValueError('Do Not Exist This Dataset')

        # Loss Function Setting
        self.loss_att = AttLoss(8.0)
        self.loss_nce = NormalizedCrossEntropy()
        self.loss_spl = CategoryCrossEntropy(self.args.T)

        # Model Setting
        self.model = WSTAL(self.args).to(self.device)
        self.memory = Memory(self.args).to(self.device)



        # Model Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model_module = self.model.module
        else:
            self.model_module = self.model



        # Loading Pretrained Model
        if self.args.pretrained and self.args.load_epoch is not None:
            model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                self.args.load_epoch) + '.pkl'
            if os.path.isfile(model_dir):
                self.model_module.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')

        parameters = [{"params": self.model.parameters()}]
        # parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, betas=[0.9, 0.99],
                                            weight_decay=self.args.weight_decay)

        # Optimizer Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.args.gpu_ids)
            self.optimizer_module = self.optimizer.module
        else:
            self.optimizer_module = self.optimizer

    def processing(self):
        if self.args.run_type == 0:
            self.train()
        elif self.args.run_type == 1:
            self.val(self.args.load_epoch)
        else:
            raise ValueError('Do not Exist This Processing')



    def train(self):
        print('Start training!')
        self.model_module.train(mode=True)

        if self.args.pretrained and self.args.load_epoch is not None:
            epoch_range = range(self.args.load_epoch, self.args.max_epoch)
        else:
            epoch_range = range(self.args.max_epoch)

        iter = 0
        step = 0
        current_lr = self.args.lr

        loss_recorder = {
            'cls_fore': 0,
            'cls_back': 0,
            'att': 0,
            'spl': 0,
        }
        criterion = WSLoss(lambda_b=self.args.back_loss_weight, lambda_att=self.args.att_loss_weight,
                           lambda_spl=self.args.spl_loss_weight, propotion=self.args.propotion, temperature=self.args.temperature,
                           weight=self.args.weight)
        # self.test(self.model, self.test_data_loader, criterion,0, phase="test")

        for epoch in epoch_range:

            for num, sample in enumerate(self.train_data_loader):
                if self.args.decay_type == 0:
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 1:
                    if num == 0:
                        current_lr = self.Step_decay_lr(epoch)
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 2:
                    current_lr = self.Cosine_decay_lr(epoch, num)
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr

                np_features = sample[0].numpy()
                np_labels = sample[1].numpy()

                labels = torch.from_numpy(np_labels).float().to(self.device)
                features = torch.from_numpy(np_features).float().to(self.device)

                f_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)
                b_labels = torch.cat([labels, torch.ones(labels.size(0), 1).to(self.device)], -1)

                o_out, m_out, em_out = self.model(features)



                if self.args.use_foreloss==1:
                    vid_fore_loss = self.loss_nce(o_out[0], f_labels) + self.loss_nce(m_out[0], f_labels)
                else:
                    vid_fore_loss = torch.tensor(0).to(self.device)

                if self.args.use_backloss==1:
                    vid_back_loss = self.loss_nce(o_out[1], b_labels) + self.loss_nce(m_out[1], b_labels)
                else:
                    vid_back_loss = torch.tensor(0).to(self.device)

                if self.args.use_attloss == 1:
                    vid_att_loss = self.loss_att(o_out[2])
                else:
                    vid_att_loss = torch.tensor(0).to(self.device)

                if epoch > self.args.warmup_epoch and self.args.use_mem==1:
                    idxs = np.where(np_labels==1)[0].tolist()
                    cls_mu = self.memory._return_queue(idxs).detach()
                    reallocated_x = random_walk(em_out[0], cls_mu, self.args.w)
                    r_vid_ca_pred, r_vid_cw_pred, r_frm_fore_att, r_frm_pred = self.model.PredictionModule(reallocated_x)

                    vid_fore_loss += 0.5 * self.loss_nce(r_vid_ca_pred, f_labels)
                    vid_back_loss += 0.5 * self.loss_nce(r_vid_cw_pred, b_labels)
                    vid_spl_loss = self.loss_spl(o_out[3], r_frm_pred * 0.2 + m_out[3] * 0.8)

                    self.memory._update_queue(em_out[1].squeeze(0), em_out[2].squeeze(0), idxs)
                else:
                    vid_spl_loss = self.loss_spl(o_out[3], m_out[3])



                total_loss = vid_fore_loss*self.args.fore_loss_weight + vid_back_loss * self.args.back_loss_weight \
                + vid_att_loss * self.args.att_loss_weight + vid_spl_loss * self.args.spl_loss_weight

                loss_recorder['cls_fore'] += vid_fore_loss.item()
                loss_recorder['cls_back'] += vid_back_loss.item()
                loss_recorder['att'] += vid_att_loss.item()
                loss_recorder['spl'] += vid_spl_loss.item()
                total_loss.backward()

                step += 1
                print('Epoch: {}/{}, Iter: {:02d}, Lr: {:.6f}'.format(
                        epoch + 1,
                        self.args.max_epoch,
                        step,
                        current_lr), end=' ')
                for k, v in loss_recorder.items():
                    print('Loss_{}: {:.4f}'.format(k, v / self.args.batch_size), end=' ')
                    loss_recorder[k] = 0

                print()
                self.optimizer_module.step()
                self.optimizer_module.zero_grad()

            if epoch == self.args.warmup_epoch:
                self.model_module.eval()
                # mu_queue, sc_queue, lbl_queue = generate_mu(self.train_data_loader_tmp, self.model_module, self.args, self.device)
                # self.memory._init_queue(mu_queue, sc_queue, lbl_queue)
                mu_queue, label_queue = generate_mu(self.args, self.model, self.train_data_loader_tmp)
                self.memory._init_queue(mu_queue, label_queue)
                self.model_module.train()
                self.args.spl_loss_weight = 0.5


            if (epoch + 1) % self.args.save_interval == 0:
                out_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                    epoch + 1) + '.pkl'
                torch.save(self.model_module.state_dict(), out_dir)
                self.model_module.eval()
                # ss_eval(epoch + 1, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
                self.test(self.model, self.test_data_loader, criterion, epoch, phase="test")
                self.model_module.train()




    def val(self, epoch):
        print('Start testing!')
        self.model_module.eval()
        ss_eval(epoch, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
        print('Finish testing!')

    def Step_decay_lr(self, epoch):
        lr_list = []
        current_epoch = epoch + 1
        for i in range(0, len(self.args.changeLR_list) + 1):
            lr_list.append(self.args.lr * ((self.args.lr_decay) ** i))

        lr_range = self.args.changeLR_list.copy()
        lr_range.insert(0, 0)
        lr_range.append(self.args.max_epoch + 1)

        if len(self.args.changeLR_list) != 0:
            for i in range(0, len(lr_range) - 1):
                if lr_range[i + 1] >= current_epoch > lr_range[i]:
                    lr_step = i
                    break

        current_lr = lr_list[lr_step]
        return current_lr

    def Cosine_decay_lr(self, epoch, batch):
        if self.args.warmup:
            max_epoch = self.args.max_epoch - self.args.warmup_epoch
            current_epoch = epoch + 1 - self.args.warmup_epoch
        else:
            max_epoch = self.args.max_epoch
            current_epoch = epoch + 1

        current_lr = 1 / 2.0 * (1.0 + np.cos(
            (current_epoch * self.args.batch_num + batch) / (max_epoch * self.args.batch_num) * np.pi)) * self.args.lr

        return current_lr

    def test(self, model, dataloader, criterion,epoch, phase="test"):

        model.eval()
        print("-------------------------------------------------------------------------------")
        device = self.device
        save_dir = os.path.join("./saves",self.args.dataset_name)
        save_dir = os.path.join(save_dir,self.args.model_id)

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
            t_factor = (self.args.segment_frames_num * vid_len) / (
                        self.args.frames_per_sec * self.args.test_upgrade_scale * input_feature.shape[1])
            # --------------------------------------------------------------------------#
            # --------------------------------------------------------------------------#
            o_out, m_out, em_out = model(input_feature)
            # --------------------------------------------------------------------------#
            # --------------------------------------------------------------------------#
            loss, loss_dict = criterion(o_out, m_out, None, vid_label_t)

            loss_stack.append(loss.cpu().item())
            fg_loss_stack.append(loss_dict["fg_loss"])
            bg_loss_stack.append(loss_dict["bg_loss"])
            att_loss_stack.append(loss_dict["att_loss"])
            spl_loss_stack.append(loss_dict["spl_loss"])
            # --------------------------------------------------------------------------#
            # --------------------------------------------------------------------------#
            o_cas = F.softmax(o_out[3], -1)
            m_cas = F.softmax(m_out[3], -1)
            temp_cas = self.args.o_weight * o_cas + self.args.m_weight * m_cas
            temp_att = o_out[2]
            test_tmp_data_log_dict[vid_name[0]] = {}
            test_tmp_data_log_dict[vid_name[0]]["vid_len"] = vid_len
            test_tmp_data_log_dict[vid_name[0]]["temp_o_cls_score_np"] = o_cas.cpu().detach().numpy()
            test_tmp_data_log_dict[vid_name[0]]["temp_m_cls_score_np"] = m_cas.cpu().detach().numpy()
            test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cas.cpu().detach().numpy()
            # --------------------------------------------------------------------------#
            # --------------------------------------------------------------------------#
            act_cls = self.args.o_weight * o_out[0] + self.args.m_weight * m_out[0]
            fg_score = act_cls[:, :self.args.action_cls_num]
            label_np = vid_label_t.cpu().detach().numpy()
            score_np = fg_score.detach().cpu().numpy()
            pred_np = np.zeros_like(score_np)
            pred_np[score_np >= self.args.cls_threshold] = 1
            pred_np[score_np < self.args.cls_threshold] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)
            test_num_correct += np.sum((correct_pred == self.args.action_cls_num))
            test_num_total += correct_pred.shape[0]
            # --------------------------------------------------------------------------#
            # --------------------------------------------------------------------------#
            # GENERATE PROPORALS.
            temp_cls_score_np = temp_cas[:, :, :self.args.action_cls_num].cpu().detach().numpy()
            temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], self.args.action_cls_num, 1))
            temp_att_score_np = temp_att.unsqueeze(2).expand([-1, -1, self.args.action_cls_num]).cpu().detach().numpy()
            temp_att_score_np = np.reshape(temp_att_score_np, (temp_cas.shape[1], self.args.action_cls_num, 1))

            score_np = np.reshape(score_np, (-1))
            if score_np.max() > self.args.cls_threshold:
                cls_prediction = np.array(np.where(score_np > self.args.cls_threshold)[0])
            else:
                cls_prediction = np.array([np.argmax(score_np)], dtype=np.int)

            temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
            temp_att_score_np = temp_att_score_np[:, cls_prediction]

            test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cls_score_np
            int_temp_cls_score_np = upgrade_resolution(temp_cls_score_np, self.args.test_upgrade_scale)
            int_temp_att_score_np = upgrade_resolution(temp_att_score_np, self.args.test_upgrade_scale)

            # int_temp_cls_score_np = (int_temp_cls_score_np - np.min(int_temp_cls_score_np, axis=0, keepdims=True)) / \
            # (np.max(int_temp_cls_score_np, axis=0, keepdims=True) - np.min(int_temp_cls_score_np, axis=0, keepdims=True) + 1e-6)
            int_temp_cls_score_np = smooth(int_temp_cls_score_np)
            # int_temp_att_score_np = smooth(int_temp_att_score_np)
            # int_temp_cls_score_np = int_temp_cls_score_np / (np.max(int_temp_cls_score_np, axis=0, keepdims=True) + 1e-6)
            # int_temp_att_score_np = int_temp_att_score_np / (np.max(int_temp_att_score_np, axis=0, keepdims=True) + 1e-6)

            cas_act_thresh = np.arange(0.001, 0.07, 0.001)
            cas_att_thresh = np.arange(0.0000, 0.001, 0.0002)

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

                props_list = get_proposal_oic(tmp_seg_list, (0.4 * tmp_int_cas + 0.6 * int_temp_att_score_np),
                                              cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.10)

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

                props_list = get_proposal_oic(tmp_seg_list, (0.3 * int_temp_cls_score_np + 0.7 * tmp_int_att),
                                              cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.250)

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
                final_proposals.append(nms(proposal_dict[class_id], self.args.nms_thresh))

            test_final_result['results'][vid_name[0]] = result2json(final_proposals, class_name_lst)

        test_acc = test_num_correct / test_num_total
        #
        # if self.args.test:
        #     # Final Test
        #     test_final_json_path = os.path.join(save_dir, "final_test_{}_result.json".format(self.args.dataset))
        # else:
        #     # Train Evalutaion
        test_final_json_path = os.path.join(save_dir, "{}_lateset_result.json".format(self.args.dataset_name))

        with open(test_final_json_path, 'w') as f:
            json.dump(test_final_result, f)

        tiou_thresholds = np.arange(0.50, 1.00, 0.05)
        anet_detection = ANETDetection(ground_truth_file=self.args.test_gt_file_path,
                                       prediction_file=test_final_json_path,
                                       tiou_thresholds=tiou_thresholds,
                                       subset="val")

        test_mAP = anet_detection.evaluate()

        # logger.log_value('Test Classification mAP', cmap, epoch)
        self.logger.add_scalar('mAP', test_acc, epoch)
        for item in list(zip(anet_detection.mAP, tiou_thresholds)):
            # logger.log_value('Test Detection1 mAP @ IoU = ' + str(item[1]), item[0], epoch)
            self.logger.add_scalar('mAP@' + str(item[1]), item[0], epoch)

        # print('average map = %f' % (sum / count))
        self.logger.add_scalar('mAP@0.1:0.7', test_mAP, epoch)
        #
        write_results_to_file(self.args, anet_detection.mAP, test_mAP, test_acc, epoch)


