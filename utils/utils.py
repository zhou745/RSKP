import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def idx2multihot(id_list, num_class):
    return np.sum(np.eye(num_class)[id_list], axis=0)


def random_choose(v_len, num_seg):
    start_ind = np.random.randint(0, v_len - num_seg)
    random_p = np.arange(start_ind, start_ind + num_seg)
    return random_p.astype(int)


def random_perturb(v_len, num_seg):
    random_p = np.arange(num_seg) * v_len / num_seg
    for i in range(num_seg):
        if i < num_seg - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(v_len, num_seg):
    u_sample = np.arange(num_seg) * v_len / num_seg
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)

def write_results_to_eval_file(args, dmap, avg, itr1, itr2):
    file_folder = './ckpt/' + args.dataset_name + '/eval/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr1)
    string_to_write += ' ' + str(itr2)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % avg    
    fid.write(string_to_write + '\n')
    fid.close()


def write_results_to_file(args, dmap, avg, cmap, itr):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % avg    
    string_to_write += ' ' + '%.2f' % cmap
    fid.write(string_to_write + '\n')
    fid.close()

def write_settings_to_file(args):
    file_folder = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/'
    file_name = args.dataset_name + '-results.log'
    fid = open(file_folder + file_name, 'a+')
    string_to_write = '#' * 80 + '\n'
    for arg in vars(args):
        string_to_write += str(arg) + ': ' + str(getattr(args, arg)) + '\n'
    string_to_write += '*' * 80 + '\n'
    fid.write(string_to_write)
    fid.close()

def compute_score_matrix(features):
    features_norm = features/features.norm(dim=-1,keepdim=True)
    scores = torch.einsum("nkd,ntd->nkt",[features_norm, features_norm])

    return(scores)

def compute_meanscore(score_matrix,s,e):
    sub_matrix = score_matrix[s:e,s:e]
    score = sub_matrix.mean().cpu().detach().numpy()
    return(score)

def compute_action_labels(scores,labels, threashhold,num_class = 20):
    segment_act = torch.zeros((scores.shape[0],scores.shape[1],num_class+1),dtype=torch.float).to(scores.device)
    for idx in range(scores.shape[0]):
        act = 0  # start from empty act
        s = 0
        label_list = torch.where(labels[idx]>0.5)[0]
        for idj in range(1, scores[idx].shape[0]):
            score = compute_meanscore(scores[idx], s, idj)
            if score < threashhold[act]:
                if act == 1:
                    for c_idx in range(label_list.shape[0]):
                        segment_act[idx,s:idj,label_list[c_idx]] = threashhold[2]/labels[idx].sum()
                    segment_act[idx, s:idj, -1] = 1-threashhold[2]
                else:
                    for c_idx in range(label_list.shape[0]):
                        segment_act[idx,s:idj,label_list[c_idx]] = (1-threashhold[3])/labels[idx].sum()
                    segment_act[idx, s:idj, -1] = threashhold[3]
                act = 1 - act
                s = idj

    return(segment_act)

def prediction_fuse_module(segs, overlapThresh, fuse_power = 7):
    # if there are no boxes, return an empty list
    if len(segs) == 0:
        return [], []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if segs.dtype.kind == "i":
        segs = segs.astype("float")

    # initialize the list of picked indexes
    pick = []
    refined_seg = []
    # grab the coordinates of the segments
    s = segs[:, 0]
    e = segs[:, 1]
    scores = segs[:, 2]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = e - s + 1
    idxs = np.argsort(scores)


    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]

        pick.append(i)

        # find the largest coordinates for the start of
        maxs = np.maximum(s[i], s[idxs[:last]])
        mine = np.minimum(e[i], e[idxs[:last]])

        # compute the length of the overlapping area
        l = np.maximum(0, mine - maxs + 1)
        # compute the ratio of overlap
        overlap = l / area[idxs[:last]]
        # compute all the position that is possibly occupied
        s_all = 0.
        e_all = 0.
        score_all = 0.
        weight_all = 0.
        for idj in np.concatenate(([last], np.where(overlap > overlapThresh)[0])):
            id_seg = idxs[idj]
            s_all += s[id_seg] * (np.sign(scores[id_seg]) * np.abs(scores[id_seg]) ** fuse_power)
            e_all += e[id_seg] * (np.sign(scores[id_seg]) * np.abs(scores[id_seg]) ** fuse_power)
            score_all += scores[id_seg] * (np.sign(scores[id_seg]) * np.abs(scores[id_seg]) ** fuse_power)
            weight_all += (np.sign(scores[id_seg]) * np.abs(scores[id_seg]) ** fuse_power)
        s_all = s_all / weight_all
        e_all = e_all / weight_all
        score_all = score_all / weight_all
        refined_seg.append([s_all, e_all, score_all])
        # delete segments beyond the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return (pick, refined_seg)

def convert_segment_to_tensor(prediction_list, vid_lens, num_cls = 21):
    pred = np.zeros([vid_lens,num_cls],dtype=float)
    #parse the label cls by cls
    for c in range(num_cls):
        segments_c = prediction_list[c]
        for seg_c in segments_c:
            pred[int(seg_c[0]):int(seg_c[1]+1),c] += seg_c[2]


    return(pred)


def generate_segs(vid_preds, frm_preds, vid_lens,gtlabels, args):

    if args.feature_type == 'UNT':
        factor = 10.0 / 4.0
    else:
        factor = 25.0 / 16.0

    #get the number of cls
    num_cls = gtlabels.shape[-1]
    act_thresh_cas = np.arange(args.start_threshold, args.end_threshold, args.threshold_interval)

    all_prediction_list = []
    for c in range(num_cls):
        c_temp = []

        vid_cls_score = vid_preds[c]
        #deter min by gt labels
        if vid_cls_score < args.class_threshold:
            all_prediction_list.append(c_temp)
            continue

        vid_cas = frm_preds[:, c]
        vid_cls_proposal = []

        for t in range(len(act_thresh_cas)):
            thres = act_thresh_cas[t]
            vid_pred = np.concatenate([np.zeros(1), (vid_cas > thres).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                len_proposal = e[j] - s[j]
                if len_proposal >= 2:
                    inner_score = np.mean(vid_cas[s[j]:e[j] + 1])
                    outer_s = max(0, int(s[j] - 0.0785 * (len_proposal ** 0.9)))
                    outer_e = min(int(vid_cas.shape[0] - 1), int(e[j] + 0.0785 * (len_proposal ** 0.9) + 1))
                    outer_temp_list = list(range(outer_s, int(s[j]))) + list(range(int(e[j] + 1), outer_e))
                    if len(outer_temp_list) == 0:
                        outer_score = 0
                    else:
                        outer_score = np.mean(vid_cas[outer_temp_list])
                    c_score = inner_score - outer_score
                    vid_cls_proposal.append([s[j], e[j] + 1, c_score])
        pick_idx, refined_seg = prediction_fuse_module(np.array(vid_cls_proposal), 0.25)

        nms_vid_cls_proposal = refined_seg
        c_temp += nms_vid_cls_proposal
        if len(c_temp) > 0:
            c_temp = np.array(c_temp)
            c_temp = c_temp[np.argsort(-c_temp[:, 2])]

        # import pdb; pdb.set_trace()
        all_prediction_list.append(c_temp)
    #convert segment prediction back to perfram labels
    return all_prediction_list

def fuse_current_pred(d_o_out, d_m_out, d_em_out,gtlabels, args, approach="naive"):

    if approach=="naive":
        frm_pred = F.softmax(d_o_out[3], -1) * args.frm_coef + F.softmax(d_m_out[3], -1) * (1 - args.frm_coef)

    vid_pred = d_o_out[0] * 0.6 + d_m_out[0] * 0.4
    vid_att = d_o_out[2]
    frm_pred = frm_pred * vid_att[..., None]
    vid_pred = np.squeeze(vid_pred.cpu().data.numpy(), axis=0)
    frm_pred = np.squeeze(frm_pred.cpu().data.numpy(), axis=0)
    vid_lens = frm_pred.shape[0]

    prediction_list = generate_segs(vid_pred, frm_pred, vid_lens,gtlabels, args)
    fram_pred = convert_segment_to_tensor(prediction_list, vid_lens)

    #back to cuda
    fram_pred_cuda = torch.tensor(fram_pred).to(d_o_out[0].device)
    return(fram_pred_cuda)

def compute_distloss(o_out, m_out, em_out, pred_label):
    loss_o = -(F.log_softmax(o_out[3], -1)*pred_label).sum(dim=-1).mean()

    return(loss_o)