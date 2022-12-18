#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necesaary Library
import scipy.io
import numpy as np
import sys
import timeit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision

from skimage.transform import radon, rescale, iradon

# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ### Loading CT Scans

# In[3]:


data = scipy.io.loadmat("ctscan_hw1.mat")
temp_ct_scans = data['ctscan']
ct_scans = []
for i in range(temp_ct_scans.shape[2]):
    ct_scans.append(temp_ct_scans[:,:,i])
ct_scans = np.array(ct_scans)


# ### Loading Infection Masks

# In[4]:


data = scipy.io.loadmat("infmsk_hw1.mat")
infmask = data['infmsk']
infection_masks = []
for i in range(infmask.shape[2]):
    infection_masks.append(infmask[:,:,i])
infection_masks = np.array(infection_masks)


# ### Data Loaders

# In[5]:


X = ct_scans
y = infection_masks

print(X.shape)
print(y.shape)
print()

X_train, X_test_temp, y_train, y_test_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_temp, y_test_temp, test_size=0.6666667, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
print()

X_train = torch.Tensor(X_train).unsqueeze(1)
y_train = torch.Tensor(y_train).unsqueeze(1)
train_dataset_tensor = TensorDataset(X_train,y_train)
train_dataloader = DataLoader(train_dataset_tensor,batch_size=32, shuffle=False)
print(len(train_dataloader))

X_val = torch.Tensor(X_val).unsqueeze(1)
y_val = torch.Tensor(y_val).unsqueeze(1)
val_dataset_tensor = TensorDataset(X_val,y_val)
val_dataloader = DataLoader(val_dataset_tensor,batch_size=32, shuffle=False)
print(len(val_dataloader))

X_test = torch.Tensor(X_test).unsqueeze(1)
y_test = torch.Tensor(y_test).unsqueeze(1)
test_dataset_tensor = TensorDataset(X_test,y_test)
test_dataloader = DataLoader(test_dataset_tensor,batch_size=32, shuffle=False)
print(len(test_dataloader))


# ## UNet Model

# ### Architecture(Pre-Trained)

# In[7]:


UNet_pretrained = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

#Changing number of channels for our task
UNet_pretrained.inc.double_conv[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
UNet_pretrained.outc.conv = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))

# UNet = UNet.to(device)

# Checking for shape fitting
# pred_masks = np.array([], dtype=np.int64).reshape(0,512,512)
# # print(pred_masks.shape)
# for batch in test_dataloader:
#     x, y = batch
#     x = x.to(device)
#     y = y.to(device)
    
#     with torch.no_grad():
#         forwardValue = UNet(x)
#         forwardValue = forwardValue.squeeze(1).detach().cpu().numpy()
#         pred_masks = np.concatenate((pred_masks,forwardValue), axis=0)
        
# print(pred_masks.shape)


# ### Architecture(Scratch)

# In[8]:


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True, out_sz=(512,512)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


# In[9]:


class SegmentationModel:
    def __init__(self, epochs, NNobj, learningRate):
        self.epochs = epochs
        self.NNobj = NNobj
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.NNobj.parameters(), lr = learningRate)

        self.trainLossList = []
        self.validLossList = []
        self.trainLossListPlotting = []
        self.validLossListPlotting = []

    def fit(self, trainData, validData):
        for i in range(self.epochs):
            self.trainLossList = []
            self.validLossList = []
            for batch in trainData:
                self.NNobj.train()
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                forwardValue = self.NNobj(x)
                costFunction_J = self.cross_entropy_loss(forwardValue, y)
                self.NNobj.zero_grad()
                costFunction_J.backward()
                self.optimizer.step()
                self.trainLossList.append(costFunction_J.item())

            for batch in validData:
                self.NNobj.eval()
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    forwardValue = self.NNobj(x)
                    costFunction_J = self.cross_entropy_loss(forwardValue, y)

                self.validLossList.append(costFunction_J.item())

            self.trainLossListPlotting.append(torch.tensor(self.trainLossList).mean())
            self.validLossListPlotting.append(torch.tensor(self.validLossList).mean())

            print('At Epoch Number: ' + str(i+1) +'; Train Loss= ' + str("{:.2f}".format(torch.tensor(self.trainLossList).mean()))+'; Validation Loss= ' + str("{:.2f}".format(torch.tensor(self.validLossList).mean())))

    def predict(self,testData):
        pred_masks = np.array([], dtype=np.int64).reshape(0,512,512)
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                forwardValue = self.NNobj(x)
                forwardValue = forwardValue.squeeze(1).detach().cpu().numpy()
                pred_masks = np.concatenate((pred_masks,forwardValue), axis=0)
        
        return pred_masks

    def plotLossCurve(self, flag):
        x = [(i+1) for i in range(self.epochs)]
        plt.xlabel('#Epoch')
        plt.ylabel('Loss')
        tempStr = "Loss curve for "
        if(flag == 0):
            tempStr += "Train Data"
            plt.plot(x,self.trainLossListPlotting)
        else:
            tempStr += "Validation Data"
            plt.plot(x,self.validLossListPlotting)
        plt.title(tempStr)
        plt.show()


# In[10]:


### Training and testing


# In[11]:


model = SegmentationModel(50, UNet().to(device), 0.001)
model.fit(train_dataloader, val_dataloader)


# In[12]:


pred_masks = model.predict(test_dataloader)


# ### Checking the Predicted Mask with the expert annotations

# In[13]:


for i in range(pred_masks.shape[0]):
    f, axarr = plt.subplots(1,3)
    axarr[0].set_title("Expert Annotations")
    axarr[0].imshow(infection_masks[2842+i])
    axarr[1].set_title("Predicted Masks")
    axarr[1].imshow(pred_masks[i])
    axarr[2].set_title("CT Scans")
    axarr[2].imshow(ct_scans[2842+i])
    plt.show()


# ### Two Samples with Expert Annotations(left), Predicted Mask(middle) and CT Scans(right)

# In[ ]:


plt.rcParams["figure.figsize"] = (12,12)
i = 99
f, axarr = plt.subplots(1,3)
axarr[0].set_title("Expert Annotations")
axarr[0].imshow(infection_masks[2842+i], cmap='gray')
axarr[1].set_title("Predicted Masks")
axarr[1].imshow(pred_masks[i], cmap='gray')
axarr[2].set_title("CT Scans")
axarr[2].imshow(ct_scans[2842+i], cmap='gray')
f.tight_layout()
plt.show()

i = 69
f, axarr = plt.subplots(1,3)
axarr[0].set_title("Expert Annotations")
axarr[0].imshow(infection_masks[2842+i], cmap='gray')
axarr[1].set_title("Predicted Masks")
axarr[1].imshow(pred_masks[i], cmap='gray')
axarr[2].set_title("CT Scans")
axarr[2].imshow(ct_scans[2842+i], cmap='gray')
f.tight_layout()
plt.show()


# ### Evaluating the model performance using several evaluation metrics

# In[ ]:


def get_confusion_metric(true_y, pred_y):
    true_y = true_y.flatten()
    pred_y = pred_y.flatten()
    return confusion_matrix(true_y, pred_y,labels=[0,1,2])
  
def get_req_avg_eval_metrics(infection_masks, pred_masks):

    avg_infection_sensitivity = 0
    avg_infection_specificity = 0
    avg_infection_accuracy = 0
    avg_infection_dice_score = 0

    avg_healthy_sensitivity = 0
    avg_healthy_specificity = 0
    avg_healthy_accuracy = 0
    avg_healthy_dice_score = 0

    count_infection_sensitivity = 0               # nan error
    N = len(infection_masks)
    for i in range(N):

        curr_confusion_metric = (get_confusion_metric(infection_masks[i],pred_masks[i])).T

        infection_TP = curr_confusion_metric[1][1]
        infection_TN = curr_confusion_metric[0][0] + curr_confusion_metric[2][0] + curr_confusion_metric[0][2] + curr_confusion_metric[2][2]
        infection_FP = curr_confusion_metric[1][0] + curr_confusion_metric[1][2] 
        infection_FN = curr_confusion_metric[0][1] + curr_confusion_metric[2][1]

        healthy_TP = curr_confusion_metric[2][2]
        healthy_TN = curr_confusion_metric[0][0] + curr_confusion_metric[0][1] + curr_confusion_metric[1][0] + curr_confusion_metric[1][1]
        healthy_FP = curr_confusion_metric[2][0] + curr_confusion_metric[2][1] 
        healthy_FN = curr_confusion_metric[0][2] + curr_confusion_metric[1][2]

        # Sensitivity = Recall = TP/(TP+FN)
        # Preicision = TP/(TP+FP)
        # Specificity = TN/(TN+FP)
        # Dice Score = 2.TP / (2.TP + FP + FN)

        infection_sensitivity = 0
        if((infection_TP+infection_FN)!=0):
            count_infection_sensitivity += 1
            infection_sensitivity = (infection_TP)/(infection_TP+infection_FN)

        infection_specificity = (infection_TN)/(infection_TN+infection_FP)
        infection_accuracy = (infection_TP+infection_TN)/(infection_TP+infection_TN+infection_FP+infection_FN)
        infection_dice_score = (2*infection_TP)/(2*infection_TP + infection_FP + infection_FN)

        healthy_sensitivity = (healthy_TP)/(healthy_TP+healthy_FN)
        healthy_specificity = (healthy_TN)/(healthy_TN+healthy_FP)
        healthy_accuracy = (healthy_TP+healthy_TN)/(healthy_TP+healthy_TN+healthy_FP+healthy_FN)
        healthy_dice_score = (2*healthy_TP)/(2*healthy_TP + healthy_FP + healthy_FN)

        avg_infection_sensitivity += infection_sensitivity
        avg_infection_specificity += infection_specificity
        avg_infection_accuracy += infection_accuracy
        avg_infection_dice_score += infection_dice_score

        avg_healthy_sensitivity += healthy_sensitivity
        avg_healthy_specificity += healthy_specificity
        avg_healthy_accuracy += healthy_accuracy
        avg_healthy_dice_score += healthy_dice_score

    avg_infection_sensitivity = avg_infection_sensitivity/count_infection_sensitivity
    avg_infection_specificity = avg_infection_specificity/N
    avg_infection_accuracy = avg_infection_accuracy/N
    avg_infection_dice_score = avg_infection_dice_score/N

    avg_healthy_sensitivity = avg_healthy_sensitivity/N
    avg_healthy_specificity = avg_healthy_specificity/N
    avg_healthy_accuracy = avg_healthy_accuracy/N
    avg_healthy_dice_score = avg_healthy_dice_score/N

    return avg_infection_dice_score, avg_infection_sensitivity, avg_infection_specificity, avg_infection_accuracy, avg_healthy_dice_score, avg_healthy_sensitivity, avg_healthy_specificity, avg_healthy_accuracy


# In[ ]:


inf_ds, inf_sen, inf_spec, inf_acc, hea_ds, hea_sen, hea_spec, hea_acc = get_req_avg_eval_metrics(infection_masks, pred_masks)
print("Average Dice Score for Infection: ", inf_ds)
print("Average Sensitivity for Infection: ", inf_sen)
print("Average Specificity for Infection: ", inf_spec)
print("Average Accuracy for Infection: ", inf_acc)
print()
print("Average Dice Score for Healthy: ", hea_ds)
print("Average Sensitivity for Healthy: ", hea_sen)
print("Average Specificity for Healthy: ", hea_spec)
print("Average Accuracy for Healthy: ", hea_acc)


# # Part B, Reconstruction
def get_predicted_mask():  
    model = SegmentationModel(50, UNet().to(device), 0.001)
    model.fit(train_dataloader, val_dataloader)
    pred_masks = model.predict(test_dataloader)
    return pred_masks



# In[8]:


def find_eval_metrics(infection_masks, pred_masks):
  inf_ds, inf_sen, inf_spec, inf_acc, hea_ds, hea_sen, hea_spec, hea_acc = get_req_avg_eval_metrics(infection_masks, pred_masks)
  print("Average Dice Score for Infection: ", inf_ds)
  print("Average Sensitivity for Infection: ", inf_sen)
  print("Average Specificity for Infection: ", inf_spec)
  print("Average Accuracy for Infection: ", inf_acc)
  print()
  print("Average Dice Score for Healthy: ", hea_ds)
  print("Average Sensitivity for Healthy: ", hea_sen)
  print("Average Specificity for Healthy: ", hea_spec)
  print("Average Accuracy for Healthy: ", hea_acc)


# In[9]:


class reconstruction_sinogram:
  def __init__(self, ct_scans):
    self.ct_scans = ct_scans
    self.sinograms = []
    self.reconstructed_ct_scans = []
    
  def get_sinogram(self, ct_scan):
    return radon(ct_scan, circle = False, preserve_range = True)

  def ct_scans_to_sinograms(self):
    N = len(self.ct_scans)
    print("CT Scans -> Sinogram")
    for i in range(N):
      sys.stdout.write('\r'+"Image No. "+str(i))
      self.sinograms.append(self.get_sinogram(self.ct_scans[i]))
    print()
    
  def get_reconstructed_ct_scan(self, sinogram, angle):
    sinogram = np.array([sinogram[:,i] for i in range(0,180, angle)])
    return iradon(sinogram.T, circle = False, preserve_range = True)

  def sinogram_to_ct_scans(self, angle):
    N = len(self.ct_scans)
    print("Sinogram -> CT Scans")
    for i in range(N):
      sys.stdout.write('\r'+"Image No. "+str(i))
      self.reconstructed_ct_scans.append(self.get_reconstructed_ct_scan(self.sinograms[i],angle))
    print()
        
  def correct_reconstruction(self,temp_infection_masks):
    N = len(self.reconstructed_ct_scans)
    for i in range(N):
      curr_inf_mask = copy.copy(temp_infection_masks[i])
      curr_inf_mask[curr_inf_mask == 2] = 1
      self.reconstructed_ct_scans[i] = np.multiply(self.reconstructed_ct_scans[i], curr_inf_mask)


# # 4x Limited Angle Sinogram

# In[10]:


reconstruct_4x = reconstruction_sinogram(ct_scans)
reconstruct_4x.ct_scans_to_sinograms()
reconstruct_4x.sinogram_to_ct_scans(angle = 4)
reconstruct_4x.correct_reconstruction(infection_masks)

# 8x Limited Angle Sinogram

# In[12]:


reconstruct_8x = reconstruction_sinogram(ct_scans)
reconstruct_8x.ct_scans_to_sinograms()
reconstruct_8x.sinogram_to_ct_scans(angle = 8)
reconstruct_8x.correct_reconstruction(infection_masks)

# Evaluating Segmentation on 4x and 8x Reconstruction

# In[15]:


pred_masks_4x = get_predicted_mask(reconstruct_4x.reconstructed_ct_scans)
print("Evaluation Metrics for 4x Reconstruction")
find_eval_metrics(infection_masks, pred_masks_4x)

print()
print()
print()

pred_masks_8x = get_predicted_mask(reconstruct_8x.reconstructed_ct_scans)
print("Evaluation Metrics for 8x Reconstruction")
find_eval_metrics(infection_masks, pred_masks_8x)
