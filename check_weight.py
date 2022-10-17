import torch
import numpy as np

def main():
    state_pretrain_1 = "./ckpt/Thumos14reduced/rskp_baseline_pretraining/200.pkl"
    state_pretrain_2 = "./ckpt/Thumos14reduced/rskp_baseline_pretraining_2/200.pkl"

    state_1 = torch.load(state_pretrain_1)
    state_2 = torch.load(state_pretrain_2)
    print("finished")

if __name__=="__main__":
    main()