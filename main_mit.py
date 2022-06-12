# python2.7
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import transforms
#from tensorboard import SummaryWriter

import os
from utils import tools
from utils import se3qua

import FlowNetC


from PIL import Image
import numpy as np

import flowlib

from PIL import Image

import csv
import time
from datasets import MITStataCenterDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import cv2

torch.cuda.empty_cache()

class MyDataset:
    
    def __init__(self, base_dir, sequence):
        self.base_dir = base_dir
        self.sequence = sequence
        self.base_path_img = self.base_dir + self.sequence + '/cam0/data/'
        
        
        self.data_files = os.listdir(self.base_dir + self.sequence + '/cam0/data/')
        self.data_files.sort()
        
        ## relative camera pose
        self.trajectory_relative = self.read_R6TrajFile('/vicon0/sampled_relative_R6.csv')
        
        ## abosolute camera pose (global)
        self.trajectory_abs = self.readTrajectoryFile('/vicon0/sampled.csv')
        
        ## imu
        self.imu = self.readIMU_File('/imu0/data.csv')
        
        self.imu_seq_len = 5
    
    def readTrajectoryFile(self, path):
        traj = []
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def read_R6TrajFile(self, path):
        traj = []
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def readIMU_File(self, path):
        imu = []
        count = 0
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if count == 0:
                    count += 1
                    continue
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                imu.append(parsed)
                
        return np.array(imu)
    
    def getTrajectoryAbs(self, idx):
        return self.trajectory_abs[idx]
    
    def getTrajectoryAbsAll(self):
        return self.trajectory_abs
    
    def getIMU(self):
        return self.imu
    
    def __len__(self):
        return len(self.trajectory_relative)
    
    def load_img_bat(self, idx, batch):
        batch_x = []
        batch_imu = []
        for i in range(batch):
            x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx + i]))
            x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1 + i]))

            ## 3 channels
            x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
            x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])

            X = np.array([x_data_np_1, x_data_np_2])
            batch_x.append(X)

            tmp = np.array(self.imu[idx-self.imu_seq_len+1 + i:idx+1 + i])
            batch_imu.append(tmp)
            
        
        batch_x = np.array(batch_x)
        batch_imu = np.array(batch_imu)
        
        X = Variable(torch.from_numpy(batch_x).type(torch.FloatTensor).cuda())    
        X2 = Variable(torch.from_numpy(batch_imu).type(torch.FloatTensor).cuda())    
        
        ## F2F gt
        Y = Variable(torch.from_numpy(self.trajectory_relative[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        ## global pose gt
        Y2 = Variable(torch.from_numpy(self.trajectory_abs[idx+1:idx+1+batch]).type(torch.FloatTensor).cuda())
        
        return X, X2, Y, Y2

    
    
class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=43014,#49165,#49152,#24576, 
            hidden_size=1024,#64, 
            num_layers=2,
            batch_first=True)
        self.rnn.cuda()
        
        self.rnnIMU = nn.LSTM(
            input_size=7,
            hidden_size=6,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 3)
        #self.linear3 = nn.Linear(128, 6)
        self.linear1.cuda()
        self.linear2.cuda()
        #self.linear3.cuda()
        
        
        
        checkpoint = None
        checkpoint_pytorch = './data/flownets/model_best.pth.tar'
        #checkpoint_pytorch = '/notebooks/data/model/FlowNet2-SD_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_pytorch):
            checkpoint = torch.load(checkpoint_pytorch,\
                                map_location=lambda storage, loc: storage.cuda(0))
            best_err = checkpoint['best_EPE']
        else:
            print('No checkpoint')

        
        self.flownet_c = FlowNetC.FlowNetC(batchNorm=False)
        self.flownet_c.load_state_dict(checkpoint['state_dict'])
        self.flownet_c.cuda()

    def forward(self, image, imu): #, xyzQ):
        batch_size, C, H, W = image.size()
        
        ## Input1: Feed image pairs to FlownetC
        c_in = image.view(batch_size, C, H, W)
        c_out = self.flownet_c(c_in)
        #print('c_out', c_out.shape)
        
        ## Input2: Feed IMU records to LSTM
        imu_out, (imu_n, imu_c) = self.rnnIMU(imu)
        imu_out = imu_out[:, -1, :]
        #print('imu_out', imu_out.shape)
        imu_out = imu_out.unsqueeze(1)
        #print('imu_out', imu_out.shape)
        
        
        ## Combine the output of input1 and 2 and feed it to LSTM
        #r_in = c_out.view(batch_size, timesteps, -1)
        r_in = c_out.view(batch_size, 1, -1)
        #print('r_in', r_in.shape)
        

        cat_out = torch.cat((r_in, imu_out), 2)#1 1 49158
        #cat_out = torch.cat((cat_out, xyzQ), 2)#1 1 49165

        #r_out, (h_n, h_c) = self.rnn(r_in)
        r_out, (h_n, h_c) = self.rnn(cat_out)
        l_out1 = self.linear1(r_out[:,-1,:])
        l_out2 = self.linear2(l_out1)
        #l_out3 = self.linear3(l_out2)

        return l_out2
    
    
def model_out_to_flow_png(output):
    out_np = output[0].data.cpu().numpy()

    #https://gitorchub.com/DediGadot/PatchBatch/blob/master/flowlib.py
    out_np = np.squeeze(out_np)
    out_np = np.moveaxis(out_np,0, -1)

    im_arr = flowlib.flow_to_image(out_np)
    im = Image.fromarray(im_arr)
    im.save('test.png')

def resizeImage(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def transformCameraData(data):
    data = resizeImage(data, 70)
    #formatted = (data * 255).astype('uint8')
    formatted = (data * 255).astype('uint8')
    img = Image.fromarray(formatted)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        #transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1]),
        #transforms.RandomCrop((320, 448)),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomHorizontalFlip()
    ])
    return preprocess(img)

def train(model, criterion, optimizer, epoch, loader):
    model.train()
    
    total_loss = 0
    n_iter = 0
    for inputs, labels in tqdm(iter(loader)):
        data = torch.cat(inputs['rgb'], 1).cuda()
        data_imu = inputs['imu'].type(torch.FloatTensor).cuda()
        labels = labels.type(torch.FloatTensor).cuda()
        
        ## Forward
        output = model(data, data_imu) #, abs_traj_input)
        
        loss = criterion(output, labels) # + criterion(abs_traj_input, target_global)
        total_loss += loss
        n_iter += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss.data.float()/n_iter

def test(model, criterion, epoch, loader):
    total_loss = 0
    n_iter = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(iter(loader)):
            data = torch.cat(inputs['rgb'], 1).cuda()
            data_imu = inputs['imu'].type(torch.FloatTensor).cuda()
            labels = labels.type(torch.FloatTensor).cuda()
            
            ## Forward
            output = model(data, data_imu) #, abs_traj_input)
            
            loss = criterion(output, labels) # + criterion(abs_traj_input, target_global)
            total_loss += loss
            n_iter += 1

    return total_loss.data.float()/n_iter
    
def main():
    batch = 1
    model = Vinet()
    #optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160, 180], gamma=0.5)
    mydataset = MITStataCenterDataset('./data/mit', '2012-04-03-07-56-24', ['rgb', 'imu'], transformCameraData, '2012-04-03-07-56-24_part4_floor2.gt.laser', False)
    train_idx, val_idx = train_test_split(list(range(len(mydataset))), test_size=0.25 , shuffle=False)

    train_data = Subset(mydataset, train_idx)
    val_data = Subset(mydataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch, shuffle=False, drop_last=True)
    #mydataset = MyDataset('../dockerData/EuRoC_modify/', 'V1_01_easy')
    #criterion  = nn.MSELoss()
    criterion  = nn.L1Loss() #size_average=False)

    for epoch in range(300):
        train_loss = train(model, criterion, optimizer, epoch, train_loader)
        val_loss = test(model, criterion, epoch, val_loader)

        #scheduler.step()
        print('Epoch {}\nTrain loss: {}\nEvaluation loss: {}\n'.format(epoch, train_loss, val_loss))
        

if __name__ == '__main__':
    main()
