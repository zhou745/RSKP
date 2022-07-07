import numpy as np
from tqdm import tqdm
from utils.utils import str2ind
import options
from evaluation.detectionMAP import getDetectionMAP as dtmAP
import copy
import torch
from model.main_branch import WSTAL
import os
import torch.nn.functional as F

f_per_feature = 16.
f_per_second = 25.

def convert_labels(gtsegments,gtlabels,features,classlist,subset):
    action_list = []
    label_list = []
    pred_segment = []
    pred_class = []
    vid_length = []
    gt_seg = []
    gt_label = []
    num_example = len(features)
    for idx in tqdm(range(num_example)):
        num_frm = features[idx].shape[0]

        action = np.array([0 for j in range(num_frm)],dtype=np.long)
        label = np.array([21 for j in range(num_frm)],dtype=np.long)
        segmet = np.array([[0 for i in range(20)]+[1] for j in range(num_frm)],dtype = np.long)
        cls = np.array([0 for j in range(20)]+[1],dtype=np.long)
        for idj in range(len(gtsegments[idx])):
            class_idx = str2ind(gtlabels[idx][idj],classlist)
            id_s = int(gtsegments[idx][idj][0]*f_per_second/f_per_feature)
            id_e = int(gtsegments[idx][idj][1]*f_per_second/f_per_feature)

            action[id_s:id_e] = 1
            label[id_s:id_e] = class_idx
            segmet[id_s:id_e,class_idx] = 1.
            segmet[id_s:id_e,-1] = 0.
            cls[class_idx]=1.
            cls[-1] = 0.
        if subset[idx]==b'test':
            pred_segment.append(segmet)
            pred_class.append(cls)
            vid_length.append(num_frm)
            gt_seg.append(gtsegments[idx])
            gt_label.append([str2ind(gtlabels[idx][idj],classlist) for idj in range(len(gtlabels[idx]))])
        action_list.append(action)
        label_list.append(label)

    return(action_list,label_list,pred_segment,pred_class,vid_length,gt_seg,gt_label)

def compute_se(actions):
    vid_pred = np.concatenate([np.zeros(1), actions, np.zeros(1)], axis=0)
    vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]

    s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
    e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
    predictions = []
    for j in range(len(s)):
        len_proposal = e[j] - s[j]
        if len_proposal>=1:
            predictions.append([s[j],e[j]])
    return(predictions)

def compute_frm_mean_std(action_list,normalize=False):

    length_list = []
    num_list = []
    num_examples = len(action_list)
    for idx in range(num_examples):
        length_frm = action_list[idx].shape[0]
        pred = compute_se(action_list[idx])
        num_frms = len(pred)

        num_list.append(num_frms)
        for idj in range(num_frms):
            id_s = pred[idj][0]
            id_e = pred[idj][1]
            length_list.append((id_e-id_s))
    length_np = np.array(length_list,dtype=float)
    num_np =  np.array(num_list,dtype=float)
    return(length_np.mean(),length_np.std(),num_np.mean(),num_np.std(),num_list)

def generate_anchor(anchor_num,type = "uniform"):
    if type =="uniform":
        return([1./(2*anchor_num)+(1.0*i)/anchor_num for i in range(anchor_num)])
    else:
        raise RuntimeError("unknown anchor type")

def genrate_psuedo_label(anchors, average_len):
    labels = []
    for i in range(len(anchors)):
        labels.append([max(anchors[i]-average_len,0.),min(anchors[i]+average_len,1.)])
    return(labels)

def match_closest(point,gt_segment):
    scores = []
    center = (point[0]+point[1])/2

    for idx in range(len(gt_segment)):
        center_seg = (gt_segment[idx][0]+gt_segment[idx][1])/2
        scores.append(np.abs(center_seg-center))
    min_value = min(scores)
    index = scores.index(min_value)
    return(index)

def compute_feature(pred_class, pred_seg, vid_length, plabels,gtseg,gtlabel):
    pred_class_new = []
    pred_seg_new = []

    for idx in range(len(pred_class)):
        pred_class_new.append(np.copy(pred_class[idx]))

        vid_len = vid_length[idx]
        #precompute the action position list
        seg = np.zeros_like(pred_seg[idx])
        seg[:,-1]=1.
        #compute the action point
        for idj in range(len(plabels)):
            s = int(plabels[idj][0]*vid_len)
            e = int(plabels[idj][1]*vid_len)
            s_t = s*f_per_feature/f_per_second
            e_t = e*f_per_feature/f_per_second

            #match the segment s,e
            index_matched = match_closest([s,e],gtseg[idx])
            seg[s:e,-1] = 0
            class_idx = gtlabel[idx][index_matched]
            seg[s:e, class_idx]=1.
        pred_seg_new.append(seg)

    return(pred_class_new,pred_seg_new)

def compute_meanscore(score_matrix,s,e):
    sub_matrix = score_matrix[s:e,s:e]
    score = sub_matrix.mean().cpu().detach().numpy()
    return(score)

def match_label(v_seg, threashhold):
    v_seg_ = torch.tensor(v_seg)
    v_seg_ = F.softmax(v_seg_*10,dim=-1)
    v_seg_norm = v_seg_/v_seg_.norm(dim=-1,keepdim=True)
    score_matrix = v_seg_norm@v_seg_norm.T
    num_frames = score_matrix.shape[0]
    segment_act = []
    act = 0  # start from empty act
    start = 0
    for idx in range(1,num_frames):
        score = compute_meanscore(score_matrix,start,idx)
        if score<threashhold[act]:
            #only act is added
            if act==1:
                segment_act.append([start,idx])
            act=1-act
            start = idx
    return(segment_act)

def compute_pred(pred_class, pred_seg, vid_length, plabels,gtseg,gtlabel):
    pred_class_new = np.copy(pred_class)
    # precompute the action position list
    pred_seg_new = np.zeros_like(pred_seg)
    pred_seg_new[:, -1] = 1.

    vid_len = vid_length

    #compute the action point
    for idj in range(len(plabels)):
        s = plabels[idj][0]
        e = plabels[idj][1]
        s_t = s*f_per_feature/f_per_second
        e_t = e*f_per_feature/f_per_second

        #match the segment s,e
        index_matched = match_closest([s,e],gtseg)
        pred_seg_new[s:e,-1] = 0
        class_idx = gtlabel[index_matched]
        pred_seg_new[s:e, class_idx]=1.
    return(pred_class_new,pred_seg_new)

def main():

    args = options.parser.parse_args()

    root = "./data/Thumos14reduced/"
    features = np.load(root+"Thumos14reduced-I3D-JOINTFeatures.npy", encoding='bytes', allow_pickle=True)
    labels = np.load(root + 'labels_all.npy', allow_pickle=True)
    classlist = np.load(root + 'classlist.npy', allow_pickle=True)
    subset = np.load(root + 'subset.npy', allow_pickle=True)
    gtsegments = np.load(root + '/segments.npy',allow_pickle=True)
    gtlabels = np.load(root + '/labels.npy', allow_pickle=True)

    action_list, label_list,pred_seg,pred_class,vid_length,gtseg, gtlabel = convert_labels(gtsegments,gtlabels,features,classlist,subset)
    #get the select list of features
    select_list = []
    for idx in range(features.shape[0]):
        if subset[idx]==b'test':
            select_list.append(1)
        else:
            select_list.append(0)
    select = np.array(select_list,dtype=np.bool)

    val_features_ori = features[select]
    #load pretrained model
    device = "cuda"
    model = WSTAL(args).to(device)
    if True:
        model_dir = './ckpt/' + args.dataset_name + '/' + str(args.model_id) + '/' + str(
            args.load_epoch) + '.pkl'
        if os.path.isfile(model_dir):
            model.load_state_dict(torch.load(model_dir))
        else:
            raise ValueError('Do Not Exist This Pretrained File')
    model.eval()
    #compute new features
    val_features = []
    val_class = []
    for idx  in tqdm(range(val_features_ori.shape[0])):
        val_features_cuda = torch.from_numpy(val_features_ori[idx]).float().to(device).unsqueeze(0)
        cls_atts = []
        with torch.no_grad():
            o_out, m_out, _ = model(val_features_cuda)
            vid_pred = o_out[0] * 0.75 + m_out[0] * 0.25
            # frm_pred = F.softmax(torch.sign(o_out[3])*torch.abs(o_out[3])**0.995, -1) * args.frm_coef + F.softmax(m_out[3], -1) * (1 - args.frm_coef)
            frm_pred = F.softmax(torch.sign(o_out[3]) * torch.abs(o_out[3]), -1) * args.frm_coef + F.softmax(
                m_out[3], -1) * (1 - args.frm_coef)
            vid_att = o_out[2]

            frm_pred = frm_pred * vid_att[..., None]
            vid_pred = np.squeeze(vid_pred.cpu().data.numpy(), axis=0)
            frm_pred = np.squeeze(frm_pred.cpu().data.numpy(), axis=0)
            val_features.append(frm_pred**0.988)
            val_class.append(vid_pred)
    val_features = np.array(val_features)
    val_class = np.array(val_class)
    #get some statistics
    # frm_mean,frm_std, num_mean, num_std, num_list = compute_frm_mean_std(action_list)
    #now we hand craft features
    anchor_num = 20
    anchor_length = 0.01

    #compute the similarity matrix
    threashold = [0.8,0.8]
    pred_class_new = []
    pred_seg_new = []
    for idx in tqdm(range(val_features_ori.shape[0])):
        seg_act = match_label(val_features[idx], threashold)
        pred_class_tmp, pred_seg_tmp = compute_pred(pred_class[idx], pred_seg[idx],vid_length[idx],seg_act,gtseg[idx],gtlabel[idx])
        pred_class_new.append(pred_class_tmp)
        pred_seg_new.append(pred_seg_tmp)
    # anchors = generate_anchor(anchor_num,type = "uniform")
    # plabels = genrate_psuedo_label(anchors,anchor_length)
    # pred_class_new,pred_seg_new = compute_feature(pred_class, pred_seg, vid_length, plabels,gtseg,gtlabel)
    #test the current prediction for psuedo label
    # dmap, iou = dtmAP(pred_class, pred_seg, vid_length, root, args)
    # dmap, iou = dtmAP(pred_class_new, pred_seg_new, vid_length, root, args)
    dmap, iou = dtmAP(val_class,val_features, vid_length, root, args)

    sum = 0
    count = 0
    for item in list(zip(iou, dmap)):
        print('Detection map @ %f = %f' % (item[0], item[1]))
        sum = sum + item[1]
        count += 1
    print('average map = %f' % (sum / count))
    print('finsished and quit')

if __name__=="__main__":
    main()