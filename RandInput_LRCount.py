#!/usr/bin/env python3
import time
import os
import shutil
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import save_image
import sys

from count_util import LinearRegionCount
from view import View



class CNN(nn.Module):
    def __init__(self,out_channel=2, k_h=1, k_w=2, stride=1, **kwargs):
        super(CNN, self).__init__()
        print('output_channel:', out_channel, '-----k_h:', k_h, '----k_w:',k_w)
        self.l1=nn.Conv2d(1, out_channel, (k_h,k_w), stride=stride, padding=0)
        self.nl1=nn.ReLU(True)
        self.l2=nn.Conv2d(out_channel, 1, (1,1), stride=1, padding=0)

    def forward(self, input):
        x=self.l1(input)
        x=self.nl1(x)
        x=self.l2(x)
        return x



class RandInput:
    def __init__(self):
        self.cfg = self.add_arguments()
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        self.model_name = self.cfg.arch + '_H' + str(self.cfg.height) + '_W' + str(self.cfg.width) + '_KH' + str(self.cfg.k_h)
        self.model_name = self.model_name + '_KW'+ str(self.cfg.k_w) + '_SD' +str(self.cfg.stride) + '_NC' + str(self.cfg.out_channel)
        self.result_path = os.path.join(self.cfg.output, self.model_name)
        os.makedirs(self.result_path, exist_ok=True)
        if self.cfg.arch == 'CNN':
            self.model = CNN(out_channel=self.cfg.out_channel, k_h=self.cfg.k_h, k_w=self.cfg.k_w, stride=self.cfg.stride)

        self.device = torch.device('cpu')

        self._initialize_weights()
        self.LRCount=LinearRegionCount()
        self.interFeature=[]
        return

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.Linear):
                n=m.weight.data.size(1)
                m.weight.data.normal_(0, math.sqrt(2./n))
                #print(m.weight.data.norm())
                m.bias.data.normal_(0, math.sqrt(2./n))
               # m.bias.data.zero_()



    def add_arguments(self):
        parser = argparse.ArgumentParser('RandInput Classification')
        model_names = ['CNN']
        parser.add_argument('-a', '--arch', metavar='ARCH', default=model_names[0], choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names))
        parser.add_argument('-out_channel', '--out_channel', type=int, default=3)
        parser.add_argument('-width', '--width', type=int, default=3)
        parser.add_argument('-height', '--height', type=int, default=1)
        parser.add_argument('-k_w', '--k_w', type=int, default=2)
        parser.add_argument('-k_h', '--k_h', type=int, default=1)
        parser.add_argument('-stride', '--stride', type=int, default=1)
        parser.add_argument('-seed', '--seed', type=int, default=1)
        parser.add_argument('-std', '--std', type=float, default=11)
        parser.add_argument('-sample_N', '--sample_N', type=int, default=2000000)
        parser.add_argument('-o', '--output', default='./results', metavar='PATH',
                           help='The root path to store results (default ./results)')
        args = parser.parse_args()
        return args

    def hook_in_forward(self, module, input, output):
        #print(module)
        self.interFeature.append(input) ## record the interFeature
        #print(len(self.interFeature))
    def train(self):
        if self.cfg.arch == 'CNN':
            print('---------train CNN---------')
            self.train_CNN(height=self.cfg.height, width=self.cfg.width, sample_number=self.cfg.sample_N, std=self.cfg.std)

        return


    def train_CNN(self, sample_number=50000, std=10, batch_size=1000, channels=1, width=2, height=2):
        print('sample_Number:', sample_number, '---std:', std, '--width:',width,'--height:', height)
        #self.sampler=torch.distributions.Uniform(-std, std)
        self.sampler=torch.distributions.Normal(0, std)
        self.model.l2.register_forward_hook(hook=self.hook_in_forward)
        for s in range(sample_number):
            end = time.time()
            inputs=torch.randn(batch_size, channels, height, width)
            numel = inputs.numel()
            rand_number=self.sampler.sample_n(numel).view_as(inputs)
            inputs = rand_number
            self.interFeature=[]
            outputs = self.model(inputs)
            feature_data=[]
            with torch.no_grad():
                #print(self.interFeature[0][0].size())
                feature_data = self.interFeature[0][0].view(batch_size,-1).data
                #feature_data = torch.cat((feature_data, self.interFeature[1][0].data), 1)
                #print(feature_data.size())
            self.LRCount.update2D(feature_data)
            LR_n= self.LRCount.getLinearReginCount()
            if s % 20 ==0:
                print('--Iter:', s, '----LR number:', LR_n,'----time cost per iteration:', time.time()-end)
        expName='SN' + str(sample_number) + '_S'+str(self.cfg.seed) + '_STD'+str(std) + '_LR' + str(LR_n)
        np.savez(os.path.join(self.result_path, expName +'.npz'),LR_n=LR_n)

        return


if __name__ == '__main__':
    Cs = RandInput()
    torch.set_num_threads(1)
    Cs.train()
