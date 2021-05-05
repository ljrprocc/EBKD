import torch
import numpy as np

if __name__ == "__main__":
    stus = ['resnet20', 'resnet20','resnet32', 'resnet8x4']
    teas = ['resnet110', 'resnet56', 'resnet110', 'resnet32x4']
    for stu, tea in zip(stus, teas):
        scores = []
        for i in range(10):
            pths = torch.load('save/student_model/S:{}_T:{}_cifar100_kd_r:0.1_a:0.9_b:0.0_{}/{}_best.pth'.format(stu, tea, i, stu))
            acc = pths['best_acc']
            scores.append(acc.item())
        scores = np.asarray(scores)
        print('T:{}, S:{}, acc:{} +- {}'.format(tea, stu, np.mean(scores), np.var(scores) ** 0.5))
