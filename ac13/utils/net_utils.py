import torch 
import random 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def weights_init(model):
    if isinstance(model, nn.Conv2d):
        model.weights.data.normal_(0.0, 0.001)
    elif isinstance(model, nn.Linear):
        model.weights.data.normal_(0.0, 0.001)

    
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