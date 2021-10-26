import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import loaddata_nyu
import util
import numpy as np
from models import modules, net, resnet, mobilenetv2
import pdb
import matplotlib
import matplotlib.image
matplotlib.rcParams['image.cmap'] = 'jet'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" ##### default gpu device

def main():
    cudnn.benchmark = True
    test_loader_nyu = loaddata_nyu.getTestingData(1)
    
    #################################  Results of the teacher ################################# 
    # Nt = net.model(modules.E_resnet(resnet.resnet34(pretrained = False)), num_features=512, block_channel = [64, 128, 256, 512])
    # Nt = torch.nn.DataParallel(Nt).cuda()
    # Nt.load_state_dict(torch.load('runs/N_t_X_U_label.pt'))     #the teacher network trained with the original labeled set and the auxiliary labeled data
    # test(test_loader_nyu, Nt)
   
    # Nt = Nt.cuda()
    # Nt.load_state_dict(torch.load('runs/N_t_X.pt'))  #the teacher network trained with the original labeled set
    # test(test_loader_nyu, Nt)

    original_model = mobilenetv2.mobilenet_v2(pretrained=False)
    if isinstance(original_model,torch.nn.DataParallel):
        original_model = original_model.module
    Ns = net.model(modules.E_mvnet2(original_model), num_features=160, block_channel = [24, 32, 96, 160])
    # print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in Ns.parameters()])))
    Ns = Ns.cuda()
    ################################### Results of the student with KD ###################################
    # Ns.load_state_dict(torch.load('runs/N_s_X_U_label.pt'))  #the student network trained with the original labeled set and the auxiliary labeled data
    # test(test_loader_nyu, Ns)

    Ns.load_state_dict(torch.load('runs/N_s_X_U_unlabel.pt')) #the student network trained with the original labeled set and the auxiliary unlabeled data
    test(test_loader_nyu, Ns)
    
    # Ns.load_state_dict(torch.load('runs/N_s_X.pt')) # the student network trained with the original labeled data
    # test(test_loader_nyu, Ns)

    # Ns.load_state_dict(torch.load('runs/N_s_U_unlabel.pt')) # the student network trained with the auxiliary unlabeled data only
    # test(test_loader_nyu, Ns)

    ################################# Results of the student without KD ###################################
    # Ns.load_state_dict(torch.load('runs/N_s_X_no_KD.pt')) # the student network trained with the original labeled set
    # test(test_loader_nyu, Ns)

    # Ns.load_state_dict(torch.load('runs/N_s_X_U_label_no_KD.pt')) # the student network trained with the original labeled set and the auxiliary labeled data 
    # test(test_loader_nyu, Ns)


def test(test_loader, model):
    model.eval()
    totalNumber = 0
    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    with torch.no_grad():
        for i, (sample_batched, flag) in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']
            image, depth = image.cuda(), depth.cuda()
             
            output = model(image)
                
            mask = (depth > 0)
            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output[mask], depth[mask])
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

            # matplotlib.image.imsave('results/out' + str(i) + '.png', output.view(output.size(2),output.size(3)).data.cpu().numpy())
        print(averageError)


if __name__ == '__main__':
    main()
