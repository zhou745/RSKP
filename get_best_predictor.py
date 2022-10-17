import numpy as np
from tqdm import tqdm
import utils.utils2 as utils
import options_pretrain2 as options
from evaluation.detectionMAP import getDetectionMAP as dtmAP
import copy
import torch
from model.main_branch import WSTAL
import os
import torch.nn.functional as F
import torch.nn as nn

f_per_feature = 16.
f_per_second = 25.
num_cls = 20

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Parameter(torch.ones(21,2048))

    def forward(self,x):
        # linear = self.linear / self.linear.norm(dim=-1, keepdim=True)
        linear = self.linear
        out = x@linear.t()
        return(out)

class svd_model(nn.Module):
    def __init__(self,centers_1,centers_2):
        super().__init__()
        U_c, S_c, V_c = torch.svd(centers_1, some=False, compute_uv=True)
        U_c_2, S_c_2, V_c_2 = torch.svd(centers_2, some=False, compute_uv=True)
        self.u = U_c
        self.v = V_c
        self.S_c = S_c

        self.u_2 = U_c_2
        self.v_2 = V_c_2
        self.S_c_2 = S_c_2

        # self.alpha = nn.Parameter(torch.tensor(1.))
        self.alpha = nn.Parameter(torch.ones((21),dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones((21), dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self,x):
        s_rescale = self.S_c*self.alpha
        classifier = self.u@(s_rescale[:,None]*self.v.t()[:21,:])

        s_rescale_2 = self.S_c_2 * self.beta
        classifier_2 = self.u_2 @ (s_rescale_2[:, None] * self.v_2.t()[:21, :])

        output = x@(classifier.t()*self.gamma+classifier_2.t()*(1-self.gamma))
        return(output)

def get_per_frame_label(gt_seg_all,gt_label_all,gt_length_all, idx_set = None,classlist = None):
    assert classlist is not None
    if idx_set is not None:
        set_idx = idx_set
    else:
        set_idx = range(len(gt_seg_all))

    gt_snippet = []
    for idx in range(len(set_idx)):
        seg_idx = set_idx[idx]

        seg_ = gt_seg_all[seg_idx]
        label_ = gt_label_all[seg_idx]
        length = gt_length_all[seg_idx]

        temp = np.zeros((length,num_cls),dtype=float)

        for seg_id in range(len(seg_)):
            s = max(int(seg_[seg_id][0]*f_per_second/f_per_feature),0)
            e = min(int(seg_[seg_id][1]*f_per_second/f_per_feature)+1,length)


            temp[s:e,utils.str2ind(label_[seg_id],classlist)] = 1.
        gt_snippet.append(temp)
    return(np.array(gt_snippet))

def compute_feature_center(feature_list, gt_list):
    value = torch.zeros((21,2048),dtype=torch.float)
    count = torch.zeros((21),dtype=torch.float)
    for idx in tqdm(range(len(feature_list))):
        feature_ = torch.tensor(feature_list[idx])
        #normalize feature
        feature = feature_/feature_.norm(dim=-1,keepdim=True)
        labels = torch.tensor(gt_list[idx])
        for c in range(21):
            if c<20:
                value[c,:] += (feature*labels[:,c].view(-1,1)).sum(dim=0)
                count[c] += labels[:,c].sum()
            else:
                value[c,:] += (feature*(1-torch.clip(labels.sum(dim=-1),max=1)).view(-1,1)).sum(dim=0)
                count[c] += (1-torch.clip(labels.sum(dim=-1),max=1)).sum()

    centers = value/(count[:,None]+1e-5)
    centers = centers/centers.norm(dim=-1,keepdim=True)
    return(centers)

def train_best_predictor(set_features_x,set_features_re,gt,device):
    model_cls = classifier().to(device)
    weight = torch.load("./centers.pkl")
    weight = weight.to(device)
    # model_cls.linear.data.copy_(weight)

    # model_cls = simple_model(predictor_ac).to(device)
    train_set_features_x_ = [torch.tensor(it) for it in set_features_x]
    train_set_features_re_ = [torch.tensor(it) for it in set_features_re]
    train_gt_ = [torch.tensor(it) for it in gt]

    train_features_cuda = torch.cat(train_set_features_x_,dim=0).to(device)
    train_features_mu_cuda = torch.cat(train_set_features_re_, dim=0).to(device)
    train_labels = torch.cat(train_gt_,dim=0).to(device)
    bg = torch.ones((train_labels.shape[0],1),dtype=torch.float32).to(device)*0.5
    train_labels_fb = torch.cat([train_labels,bg],dim=-1).argmax(dim=-1)
    optimizer = torch.optim.Adam(model_cls.parameters(),lr = 1e-5)

    for iter in range(100000):
        pred_x = model_cls(train_features_cuda)
        pred_mu = model_cls(train_features_mu_cuda)
        loss_x = nn.CrossEntropyLoss()(pred_x,train_labels_fb)
        loss_mu = nn.CrossEntropyLoss()(pred_mu, train_labels_fb)

        loss = loss_x*0.85+loss_mu*0.15
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p = F.softmax(pred_x)*0.85+F.softmax(pred_mu)*0.15
        acc = (p.argmax(dim=-1)==train_labels_fb).to(torch.float32).mean()

        if iter%100==0:
            print("acc is %f, loss is %f"%(acc.item(),loss.item()))
    print("finshed training")
    torch.save(model_cls.linear.data.clone(),"centers.pkl")
    return(weight)

def main():
    args = options.parser.parse_args()
    device = torch.device("cuda")

    root = "./data/Thumos14reduced/"
    features = np.load(root+"Thumos14reduced-I3D-JOINTFeatures.npy", encoding='bytes', allow_pickle=True)
    labels = np.load(root + 'labels_all.npy', allow_pickle=True)
    classlist = np.load(root + 'classlist.npy', allow_pickle=True)
    subset = np.load(root + 'subset.npy', allow_pickle=True)
    gtsegments = np.load(root + '/segments.npy',allow_pickle=True)
    gtlabels = np.load(root + '/labels.npy', allow_pickle=True)

    ckpt_path = "./ckpt/Thumos14reduced/rskp_baseline_pretraining_2/200.pkl"
    model_used = WSTAL(args).to(device)

    statedict = torch.load(ckpt_path)
    model_used.load_state_dict(statedict)
    #get the split idx
    trainidx = []
    testidx = []

    labels_all = np.stack([utils.strlist2multihot(labs, classlist) for labs in labels],axis=0)
    vid_len = [len(features[idx]) for idx in range(len(features))]
    for i, s in enumerate(subset):
        if s.decode('utf-8') == 'validation':  # Specific to Thumos14
            trainidx.append(i)
        else:
            testidx.append(i)

    #get the train set feature matrix
    train_features = features[trainidx]
    train_labels = labels_all[trainidx]

    train_gt = get_per_frame_label(gtsegments,gtlabels,vid_len,idx_set=trainidx,classlist=classlist)

    #get the val set feature matrix
    val_features = features[testidx]
    val_labels = labels_all[trainidx]
    val_gt = get_per_frame_label(gtsegments,gtlabels,vid_len,idx_set=testidx,classlist=classlist)

    #compute the prediction features:
    predictor_ac = model_used.ac_center.data.clone()
    predictor_fg = model_used.fg_center.data.clone()
    model_used.eval()

    val_set_features_x = []
    val_set_features_re = []
    for idx in tqdm(range(len(val_features))):
        feature = torch.tensor(val_features[idx]).to(device).unsqueeze(0)
        o_out, m_out, em_out = model_used(feature)
        val_set_features_x.append(em_out[0][0].detach().cpu().numpy())
        val_set_features_re.append(em_out[-1][0].detach().cpu().numpy())
    print("compute val feature finished")

    #compute the weight that bestly classify the val set
    train_set_features_x = []
    train_set_features_re = []
    for idx in tqdm(range(len(train_features))):
        feature = torch.tensor(train_features[idx]).to(device).unsqueeze(0)
        o_out, m_out, em_out = model_used(feature)
        train_set_features_x.append(em_out[0][0].detach().cpu().numpy())
        train_set_features_re.append(em_out[-1][0].detach().cpu().numpy())
    print("compute train feature finished")

    #compute feature centers
    # original_train_centers = compute_feature_center(train_features,train_gt)
    #
    # original_val_centers = compute_feature_center(val_features,val_gt)
    #
    # print("original center computed")
    #
    # s_ori = torch.softmax(original_train_centers@original_val_centers.t()/0.01,dim=-1)
    #
    # rskp_train_centers = compute_feature_center(train_set_features_x,train_gt)
    #
    # rskp_val_centers = compute_feature_center(val_set_features_x,val_gt)
    #
    # s_rskp = torch.softmax(rskp_train_centers @ rskp_val_centers.t() / 0.01, dim=-1)
    #
    # s_pred = torch.softmax((predictor_ac.to('cpu')/predictor_ac.to('cpu').norm(dim=-1,keepdim=True))@rskp_val_centers.t()/0.01,dim=-1)
    #
    # print("rskp center computed")
    #train the best center
    #create training data
    # weight = train_best_predictor(val_set_features_x,val_set_features_re,val_gt,device)

    #compute how do we transform the acenters to weight
    #predictor_ac
    #weight
    # weight = torch.load("centers.pkl")

    length_ac = predictor_ac.norm(dim=-1)
    ac_norm = predictor_ac/length_ac[:,None]

    # length_weight = weight.norm(dim=-1)
    # weight_norm = weight/length_weight[:,None]
    train_set_features_x_ = [torch.tensor(it) for it in train_set_features_x]
    train_set_features_re_ = [torch.tensor(it) for it in train_set_features_re]
    #compute the image label level centers
    feature_x = torch.cat(train_set_features_x_).to(device)
    feature_re = torch.cat(train_set_features_re_).to(device)
    transform_1 = ac_norm
    transform_2 = ac_norm
    scores = F.softmax(feature_x@transform_1.t()*20)*0.85+F.softmax(feature_re@transform_2.t()*20)*0.15

    scores = (scores.clip(min=0.2,max=1.0)-0.2)/0.8
    recenter = scores.t()@(feature_x*0.85+feature_re*0.15)
    recenter_norm = recenter/recenter.norm(dim=-1,keepdim=True)


    # s = weight_norm@recenter_norm.t()
    # s_weight = weight_norm@weight_norm.t()
    s_f = recenter_norm@recenter_norm.t()

    # U_norm, S_norm, V_norm = torch.svd(weight_norm, some=False, compute_uv=True)
    # U_c, S_c, V_c = torch.svd(recenter_norm, some=False, compute_uv=True)

    torch.save(recenter_norm,"centers_wk.pkl")
    torch.save(ac_norm, "centers_ac.pkl")
    #train a transofrom for thesvd decomp
    # model_svd = svd_model(recenter_norm).to(device)
    model_svd = svd_model(ac_norm,recenter_norm).to(device)
    # rescales = torch.load("rescale.pkl").to(device)
    # model_svd.alpha.data.copy_(rescales)

    val_set_features_x_ = [torch.tensor(it) for it in val_set_features_x]
    val_set_features_re_ = [torch.tensor(it) for it in val_set_features_re]
    val_gt_ = [torch.tensor(it) for it in val_gt]
    #compute the image label level centers
    feature_x_val = torch.cat(val_set_features_x_).to(device)
    feature_re_val = torch.cat(val_set_features_re_).to(device)
    val_labels = torch.cat(val_gt_, dim=0).to(device)
    bg = torch.ones((val_labels.shape[0], 1), dtype=torch.float32).to(device) * 0.5
    val_labels_fb = torch.cat([val_labels, bg], dim=-1).argmax(dim=-1)
    optimizer = torch.optim.Adam(model_svd.parameters(), lr=1e-4)
    for iter in range(1000000):
        pred_x = model_svd(feature_x_val)
        pred_mu = model_svd(feature_re_val)
        loss_x = nn.CrossEntropyLoss()(pred_x, val_labels_fb)
        loss_mu = nn.CrossEntropyLoss()(pred_mu, val_labels_fb)

        loss = loss_x * 0.85 + loss_mu * 0.15
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p = F.softmax(pred_x) * 0.85 + F.softmax(pred_mu) * 0.15
        acc = (p.argmax(dim=-1) == val_labels_fb).to(torch.float32).mean()

        if iter % 100 == 0:
            print("acc is %f, loss is %f" % (acc.item(), loss.item()))

    # torch.save(model_svd.alpha.data.clone(), "rescale.pkl")
    torch.save(model_svd.state_dict(), "svd_para.pth")
    print("finshed training")





if __name__=="__main__":
    main()