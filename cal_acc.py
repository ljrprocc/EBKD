import torch
import numpy as np
import sys

if __name__ == "__main__":
    stus = sys.argv[1].split(',')
    teas = sys.argv[2].split(',')
    logo = sys.argv[3]
    for stu, tea in zip(stus, teas):
        scores = []
        for i in range(10):
            pths = torch.load('save/student_model/S:{}_T:{}_cifar100_attention_r:1_a:0.0_b:1000.0_{}_{}_ebm/{}_best.pth'.format(stu, tea, i, logo, stu))
            acc = pths['best_acc']
            scores.append(acc.item())
        scores = np.asarray(scores)
        print('T:{}, S:{}, acc:{} +- {}'.format(tea, stu, np.mean(scores), np.var(scores) ** 0.5))
