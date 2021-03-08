#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('run', '/Desktop/Share/CUDA_DEVICE_setup.py -n 1')

import sys
import torch
from A_speech_command_dataset import SpeechCommandDataset
from module import QuatSincConv1d
from module import SincConv1d
from module import Quantize, QuaternaryConv1d, QuaternaryLinear
from module import DistrLoss
import soundfile as sf
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
print(torch.__version__)
from torchsummary import summary
import IPython


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


class LogAbs(nn.Module):
    def __init__(self):
        super(LogAbs, self).__init__()

    def forward(self, input):
        return torch.log10(torch.abs(input) + 1)


class _Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pool_size, use_sinc, cal_distrloss=True):
        super(_Layer, self).__init__()

        self.use_sinc = use_sinc
        self.cal_distrloss = cal_distrloss

        if use_sinc:
            
            self.conv = QuatSincConv1d(
                out_channels=out_channels, kernel_size=kernel_size)
            self.bn = nn.BatchNorm1d(out_channels)
            self.htanh = nn.Hardtanh(min_val=0.0)
            self.quan = Quantize(8)
        else:

            self.conv = QuaternaryConv1d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, num_of_bits=8)
            self.bn = nn.BatchNorm1d(out_channels)
            self.htanh = nn.Hardtanh(min_val=0.0)
            self.quan = Quantize(8)

        self.pool = nn.MaxPool1d(max_pool_size)

        if self.cal_distrloss:
            self.distrloss = DistrLoss(out_channels)

    def forward(self, input):
        if self.use_sinc:
            out = self.conv(input)
            out = self.bn(out)

        else:
            out = self.conv(input)
            out = self.bn(out)

        if self.cal_distrloss:
            dloss = self.distrloss(out)

        out = self.htanh(out)
        out = self.quan(out)

        out = self.pool(out)

        if self.cal_distrloss:
            return out, dloss
        else:
            return out


class QuatenarySincNet(nn.Module):
    def __init__(self, expansion=1.0, cal_distrloss=True):
        super(QuatenarySincNet, self).__init__()

        self.cal_distrloss = cal_distrloss

        self.sincconv = _Layer(
            in_channels=1, out_channels=int(8), kernel_size=50, stride=1, max_pool_size=2, use_sinc=True, cal_distrloss=cal_distrloss)

        self.features = nn.ModuleList()

        self.features.append(_Layer(in_channels=8, out_channels=int(16*expansion),
                                    kernel_size=25, stride=2, max_pool_size=10, use_sinc=False, cal_distrloss=cal_distrloss))

        self.features.append(_Layer(in_channels=int(16*expansion), out_channels=int(32*expansion),
                                    kernel_size=9, stride=1, max_pool_size=10, use_sinc=False, cal_distrloss=cal_distrloss))

        self.features.append(_Layer(in_channels=int(32*expansion), out_channels=int(64*expansion),
                                    kernel_size=9, stride=1, max_pool_size=8, use_sinc=False, cal_distrloss=cal_distrloss))

        self.gap = nn.Flatten()
        self.quan_gap = Quantize(num_of_bits=8)
        self.fc1 = nn.Linear(int(64*3*expansion), 64, bias=True)
        self.quan_fc1_w = Quantize(num_of_bits=8)
        
        self.fc2 = nn.Linear(int(64*expansion), 35, bias=True)
        self.quan_fc2_w = Quantize(num_of_bits=8)
        
        self.outs = {}

    def forward(self, input):
        dlosses = []

        if self.cal_distrloss:
            out, dloss = self.sincconv(input)
            dlosses.append(dloss)
            self.outs['sincconv'] = out[0].detach().cpu().numpy()
        else:
            out = self.sincconv(input)

        for i, l in enumerate(self.features):
            if self.cal_distrloss:
                out, dloss = l(out)
                dlosses.append(dloss)
                self.outs['conv'+str(i)] = out[0].detach().cpu().numpy()
            else:
                out = l(out)

        out = self.gap(out)
        out = self.quan_gap(out)
        out = out.view(out.size(0), -1)

        self.outs['gap'] = out[0].detach().cpu().numpy()

        self.fc1.weight.data = F.hardtanh(self.fc1.weight.data)
        self.fc1.weight.data = self.quan_fc1_w(self.fc1.weight.data)
        self.fc1.bias.data = F.hardtanh(self.fc1.bias.data)
        self.fc1.bias.data = self.quan_fc1_w(self.fc1.bias.data)
        out = self.fc1(out)
        
        self.fc2.weight.data = F.hardtanh(self.fc2.weight.data)
        self.fc2.weight.data = self.quan_fc2_w(self.fc2.weight.data)
        self.fc2.bias.data = F.hardtanh(self.fc2.bias.data)
        self.fc2.bias.data = self.quan_fc2_w(self.fc2.bias.data)
        out = self.fc2(out)
        self.outs['fc2'] = out[0].detach().cpu().numpy()

        if self.cal_distrloss:
            distrloss1 = sum([ele[0] for ele in dlosses]) / len(dlosses)
            distrloss2 = sum([ele[1] for ele in dlosses]) / len(dlosses)

            return out, distrloss1.view(1, 1), distrloss2.view(1, 1)
        else:
            return out

model = QuatenarySincNet().to(device)


# In[5]:


model = torch.load("chi_1124_Q_pruned_20.pth").to('cpu')


# In[ ]:





# In[6]:


def test123(model,path):
    audio, sr = sf.read(path)
    audio = audio.astype(np.float32)
    # padding or truncating to 1 second
    if len(audio) < sr:
        padding_size = (sr-len(audio))//2
        if len(audio) % 2 == 0:
            audio = np.pad(audio, (padding_size, padding_size),
                           mode='constant')
        else:
            audio = np.pad(audio, (padding_size, padding_size+1),
                           mode='constant')
    elif len(audio) > sr:
        truncating_size = (len(audio)-sr)//2
        if len(audio) % 2 == 0:
            audio = audio[truncating_size:truncating_size+sr]
        else:
            audio = audio[truncating_size+1:truncating_size+1+sr]
    audio = audio[np.newaxis, ...]
    audio = audio.reshape(1,1,16000)
    data = torch.tensor(audio)
    data = data.to('cpu')
    output, loss1, loss2 = model(data)
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    arr = pred.data.cpu().numpy()
    label = {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'10',10:'up',11:'down',12:'open',13:'close'}
    arr = label[arr[0][0]]
    return arr


# In[7]:


def demo(model, file):
    a = test123(model, "/Desktop/CodeFolder/KWS/1216_demo/"+file)
    print(a)


# In[8]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/a.wav")


# In[9]:


demo(model, "a.wav")


# In[10]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/b.wav")


# In[11]:


demo(model, "b.wav")


# In[12]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/c.wav")


# In[13]:


demo(model, "c.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/d.wav")


# In[ ]:


demo(model, "d.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/e.wav")


# In[ ]:


demo(model, "e.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/f.wav")


# In[14]:


demo(model, "f.wav")


# In[15]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/g.wav")


# In[16]:


demo(model, "g.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/h.wav")


# In[ ]:


demo(model, "h.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/i.wav")


# In[ ]:


demo(model, "i.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/j.wav")


# In[ ]:


demo(model, "j.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/k.wav")


# In[ ]:


demo(model, "k.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/l.wav")


# In[ ]:


demo(model, "l.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/m.wav")


# In[ ]:


demo(model, "m.wav")


# In[ ]:


IPython.display.Audio("/Desktop/CodeFolder/KWS/1216_demo/n.wav")


# In[ ]:


demo(model, "n.wav")


# In[ ]:




