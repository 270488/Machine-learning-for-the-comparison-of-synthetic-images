# Imports

# PyTorch
from unittest.mock import patch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Variable
import torchvision.transforms as tr
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as tr

# Models
from fresunet_strategy4 import FresUNet


# Other
import os
import numpy as np
import random
from skimage import io
import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm as tqdm
from math import  ceil
from IPython import display
import time

import time
import warnings

print('IMPORTS OK')

# Global Variables' Definitions

N_FEATURES=1

TRESHOLD=0.5

PATH_TO_DATASET = 'Train_Dataset/segmentation_dataset/'

IS_PROTOTYPE = False

FP_MODIFIER = 1 # Tuning parameter, use 1 if unsure 
BATCH_SIZE = 128
PATCH_SIDE = 64
N_EPOCHS =30
ALPHA=0.1
img_height, img_width=512, 512

NORMALISE_IMGS = True

TRAIN_STRIDE = int(PATCH_SIDE /2) - 1

CUDA_LAUNCH_BLOCKING=1

TYPE = 0 # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

LOAD_TRAINED = False

DATA_AUG = True

class_index=["shadow", "transparency", "texture"]

labels_train=[ "geometria",
"cuffie","bottiglia",  "vino","televisione", "palla", "ombrello", "carte", "caffe", "lampada", "donuts", "libro", "lego"]
labels_test=["forbici", "crema", "scatole","suzanne"]
labels_val=["martello", "umanoide", "scarpe" ,"strumenti"]
pov=["Top"]#, "Side", "Side2", "Front", "Perspective"]


print('DEFINITIONS OK')


    
from skimage import transform
def read_sentinel_img(path):
    img = io.imread(path)

    
    # Check if the image has an alpha channel and remove it if it does
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = transform.resize(img, (img_height, img_width), anti_aliasing=True)

    # Converti in float se necessario e normalizza
    I = img.astype('float32') 
    
    # Extract RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

  

    
    I = np.stack((r,g,b),axis=2).astype('float')
    
    
    if NORMALISE_IMGS:
        I = (I/255) 
    
    return I

def read_sentinel_img_trio_cm(path, obj, pov):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    
    
    I1 = read_sentinel_img(path+"/Reference/"+obj+"_"+pov+".png") #REFERENCE
    I2 = read_sentinel_img(path+"/Render/"+obj+"_"+pov+".png") #Render
    
    
    
    cm_shadow = io.imread(path+"/"+"Change map_shadow/"+obj+"_"+pov+".png", as_gray=True)
    cm_shadow= transform.resize(cm_shadow, (img_height, img_width))
    cm_shadow=cm_shadow!=0

    cm_transparency=io.imread(path+"/"+"Change map_transparency/"+obj+"_"+pov+".png", as_gray=True)
    cm_transparency = transform.resize(cm_transparency, (img_height, img_width)) 
    cm_transparency=cm_transparency!=0
    
    cm_texture=io.imread(path+"/"+"Change map_texture/"+obj+"_"+pov+".png", as_gray=True)
    cm_texture = transform.resize(cm_texture, (img_height, img_width)) 
    cm_texture=cm_texture!=0    

    



    cm=np.stack((cm_shadow,  cm_transparency, cm_texture), axis=2)
    cm = np.transpose(cm, (2, 0, 1)) 
   
    cm = torch.from_numpy(cm).float() 

   
    
    #cm = np.argmax(cm, axis=2)
    return I1, I2, cm

def read_sentinel_img_trio_seg(path, obj, pov, ref_ren):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    
    seg_shadow = io.imread(path+"/"+ref_ren+"_shadow/"+obj+"_"+pov+".png", as_gray=True)
    seg_shadow= transform.resize(seg_shadow, (img_height, img_width))
    seg_shadow=seg_shadow!=0

    seg_transparency=io.imread(path+"/"+ref_ren+"_transparency/"+obj+"_"+pov+".png", as_gray=True)
    seg_transparency = transform.resize(seg_transparency, (img_height, img_width)) 
    seg_transparency=seg_transparency!=0
    
    seg_texture=io.imread(path+"/"+ref_ren+"_texture/"+obj+"_"+pov+".png", as_gray=True)
    seg_texture = transform.resize(seg_texture, (img_height, img_width)) 
    seg_texture=seg_texture!=0    

    



    seg=np.stack((seg_shadow,  seg_transparency, seg_texture), axis=2)
    seg = np.transpose(seg, (2, 0, 1)) 
   
    seg = torch.from_numpy(seg).float() 

   
    
    #cm = np.argmax(cm, axis=2)
    return seg

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
#     out = np.swapaxes(I,1,2)
#     out = np.swapaxes(out,0,1)
#     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)




class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train = True, patch_side = 96, stride = None, use_all_bands = False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        self.path = path  #PATH TO DATASET
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        if train:
            fname = 'train.txt'
            self.names = labels_train #labels sono gli oggetti
        else:
            fname = 'val.txt'
            self.names = labels_val #labels sono gli oggetti
        
        
        self.point_of_view=pov    #pov sono i punti di vista

        self.n_imgs = len(self.names)
        
        n_pix = 0

        true_pix_shadow_CM = 0
        true_pix_texture_CM = 0
        true_pix_reflections_CM = 0
        true_pix_transparency_CM = 0
        
        true_pix_shadow_LCM = 0
        true_pix_texture_LCM = 0
        true_pix_reflections_LCM = 0
        true_pix_transparency_LCM = 0

        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.seg_reference = {}
        self.seg_render = {}
        
       
        self.change_map = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for obj_img in tqdm(self.names):
            for point_of_view in tqdm(self.point_of_view):
              
                    # load and store each image
                I1, I2, cm = read_sentinel_img_trio_cm(self.path + obj_img+"/", obj_img, point_of_view) #entra nella cartella dell'oggetto
                seg_reference=read_sentinel_img_trio_seg(self.path + obj_img+"/", obj_img, point_of_view, ref_ren="Reference")
                seg_render=read_sentinel_img_trio_seg(self.path + obj_img+"/", obj_img, point_of_view, ref_ren="Render")
                self.imgs_1[obj_img+"_"+point_of_view] = reshape_for_torch(I1)
                self.imgs_2[obj_img+"_"+point_of_view] = reshape_for_torch(I2)
                self.seg_render[obj_img+"_"+point_of_view] = seg_render
                self.seg_reference[obj_img+"_"+point_of_view] = seg_reference
                
                
                self.change_map[obj_img+"_"+point_of_view] = cm
                
                s = cm.shape
                n_pix += np.prod(s)
                true_pix_shadow_CM += cm[0].sum() #0 ombre, 1 trasparenze, 2 texture
                
                true_pix_transparency_CM+=cm[1].sum()
                true_pix_texture_CM+=cm[2].sum()
                
                true_pix_shadow_LCM += (seg_render[0].sum()+seg_reference[0].sum())/2 #0 ombre, 1 trasparenze, 2 texture
                
                true_pix_transparency_LCM+=(seg_render[1].sum()+seg_reference[1].sum())/2
                true_pix_texture_LCM+=(seg_render[2].sum()+seg_reference[2].sum())/2
                
                
                # calculate the number of patches
                s = self.imgs_1[obj_img+"_"+point_of_view].shape
                n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
                n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
                n_patches_i = n1 * n2
                self.n_patches_per_image[obj_img+"_"+point_of_view] = n_patches_i
                self.n_patches += n_patches_i
                
                # generate path coordinates
                for i in range(n1):
                    for j in range(n2):
                        # coordinates in (x1, x2, y1, y2)
                        current_patch_coords = (obj_img+"_"+point_of_view, 
                                        [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                        [self.stride*(i + 1), self.stride*(j + 1)])
                        self.patch_coords.append(current_patch_coords)
        
                   

        n_pix=n_pix/N_FEATURES
        self.shadow_rate_CM=(n_pix-true_pix_shadow_CM)/true_pix_shadow_CM if true_pix_shadow_CM!=0 else 0
        self.texture_rate_CM=(n_pix-true_pix_texture_CM)/true_pix_texture_CM if true_pix_texture_CM!=0 else 0
        self.transparency_rate_CM=(n_pix-true_pix_transparency_CM)/true_pix_transparency_CM if true_pix_transparency_CM!=0 else 0
        
        self.shadow_rate_LCM=(n_pix-true_pix_shadow_LCM)/true_pix_shadow_LCM if true_pix_shadow_LCM!=0 else 0
        self.texture_rate_LCM=(n_pix-true_pix_texture_LCM)/true_pix_texture_LCM if true_pix_texture_LCM!=0 else 0
        self.transparency_rate_LCM=(n_pix-true_pix_transparency_LCM)/true_pix_transparency_LCM if true_pix_transparency_LCM!=0 else 0
    
    def get_trueRate(self):
        return {'shadow_CD':self.shadow_rate_CM,
                "transparency_CD": self.transparency_rate_CM, "texture_CD": self.texture_rate_CM, 
                "shadow_LCM": self.shadow_rate_LCM, "transparency_LCM": self.transparency_rate_LCM, "texture_LCM": self.transparency_rate_LCM}
    

   

        
        
        

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name],self.change_map[im_name], self.seg_reference[im_name], self.seg_render[im_name]
   

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        
        
        label = self.change_map[im_name][:,limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()

        segmentation_reference=self.seg_reference[im_name][:,limits[0]:limits[1], limits[2]:limits[3]]
        segmentation_reference = torch.from_numpy(1*np.array(segmentation_reference)).float() 
        
        segmentation_render=self.seg_render[im_name][:,limits[0]:limits[1], limits[2]:limits[3]]
        segmentation_render = torch.from_numpy(1*np.array(segmentation_render)).float()

        
        sample = {'I1': I1, "I2": I2, 'change map': label, 'segmentation reference': segmentation_reference, 'segmentation render': segmentation_render}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class ChangeDetectionTestDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train = False, patch_side = 96, stride = None, use_all_bands = False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        self.path = path  #PATH TO DATASET
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        self.names = labels_test #labels sono gli oggetti
        
#         print(path + fname)
        #self.names = read_csv(path + fname).columns
        
        self.point_of_view=pov    #pov sono i punti di vista

        self.n_imgs = len(self.names)
      
        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        
        
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for obj_img in tqdm(self.names):
            
            for point_of_view in tqdm(self.point_of_view):
                # load and store each image
                
                I1,I2, _ = read_sentinel_img_trio_cm(self.path + obj_img+"/", obj_img, point_of_view) #entra nella cartella dell'oggetto
                self.imgs_1[obj_img+"_"+point_of_view] = reshape_for_torch(I1)
                self.imgs_2[obj_img+"_"+point_of_view] = reshape_for_torch(I2)
                
                
                
                s = I1.shape
            
                # calculate the number of patches
                s = self.imgs_1[obj_img+"_"+point_of_view].shape
                n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
                n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
                n_patches_i = n1 * n2
                self.n_patches_per_image[obj_img+"_"+point_of_view] = n_patches_i
                self.n_patches += n_patches_i
                
                # generate path coordinates
                for i in range(n1):
                    for j in range(n2):
                        # coordinates in (x1, x2, y1, y2)
                        current_patch_coords = (obj_img+"_"+point_of_view, 
                                        [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                        [self.stride*(i + 1), self.stride*(j + 1)])
                        self.patch_coords.append(current_patch_coords)
            
        
        

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name]
   

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
      
        
        
        sample = {'I1': I1, 'I2':I2}
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        I1,I2, label, seg_ref, seg_ren = sample['I1'], sample['I2'],sample['change map'], sample['segmentation reference'], sample['segmentation render']
        imgs=[I1, I2]
        
        if random.random() > 0.1:
            for i, _ in enumerate(imgs):
                imgs[i]=imgs[i].numpy()[:,:,::-1].copy()
                imgs[i]=torch.from_numpy(imgs[i])

            
            label=label.numpy()[:,::-1].copy()
            label=torch.from_numpy(label)
            
            seg_ref=seg_ref.numpy()[:,::-1].copy()
            seg_ref=torch.from_numpy(seg_ref)
            
            seg_ren=seg_ren.numpy()[:,::-1].copy()
            seg_ren=torch.from_numpy(seg_ren)
            
           
            
        return {'I1': imgs[0],'I2':imgs[1],  'change map': label, 'segmentation reference': seg_ref, 'segmentation render': seg_ren}



class RandomRot(object):
    """Rotate randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        I1,I2, label, seg_ref, seg_ren = sample['I1'], sample['I2'],sample['change map'], sample['segmentation reference'], sample['segmentation render']

        imgs=[I1, I2]
        
        n = random.randint(0, 3)
        if n:
            for i,_ in enumerate(imgs):
                imgs[i]=imgs[i].numpy()
                imgs[i]=np.rot90(imgs[i], n, axes=(1,2)).copy()
                imgs[i]=torch.from_numpy(imgs[i])
            
           
            label=label.numpy()
            label = np.rot90(label, n, axes=(1,2)).copy()
            label= torch.from_numpy(label)
            
            seg_ref=seg_ref.numpy()
            seg_ref = np.rot90(seg_ref, n, axes=(1,2)).copy()
            seg_ref= torch.from_numpy(seg_ref)
            
            seg_ren=seg_ren.numpy()
            seg_ren = np.rot90(seg_ren, n, axes=(1,2)).copy()
            seg_ren= torch.from_numpy(seg_ren)

        return {'I1': imgs[0],'I2':imgs[1],  'change map': label, 'segmentation reference': seg_ref, 'segmentation render': seg_ren}
def jaccard_loss(pred, target, weights=None, smooth=1e-6):
    # Pred deve essere un tensor di probabilità (softmax applicato)
    pred=torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))  # Somma su H e W
    union = (pred + target).sum(dim=(2, 3)) - intersection  # Unione
    
    iou = (intersection + smooth) / (union + smooth)  # Calcola IoU per ciascuna classe

    if weights is not None:
        weights = weights.view(1, -1)  # Reshape weights to broadcast across batches
        
        iou = iou * weights  # Apply weights to each class IoU
    
        # Compute weighted average IoU over classes and batches
        weighted_iou = iou.mean()
        
        # Return the weighted Jaccard loss
        return 1 - weighted_iou


def combined_loss_CD(pred, target, weights=None):
      
      # Funzione di perdita principale
    loss_BCEWithLogits=criterion_CD(pred, target)
    jaccard = jaccard_loss(pred, target, weights=weights)  # Funzione di perdita IoU
    return loss_BCEWithLogits + jaccard
def combined_loss_LCM(pred, target, weights=None):
      
      # Funzione di perdita principale
    loss_BCEWithLogits=criterion_LCM(pred, target)
    jaccard = jaccard_loss(pred, target, weights=weights)   # Funzione di perdita IoU
    return loss_BCEWithLogits + jaccard
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



print('UTILS OK')

# Dataset


if DATA_AUG:
    data_transform = tr.Compose([RandomFlip(), RandomRot()])
else:
    data_transform = None


        
train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = True, stride = TRAIN_STRIDE,patch_side=PATCH_SIDE, transform=data_transform)

#train_sampler = WeightedRandomSampler(train_dataset.sample_weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



val_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = False, patch_side=PATCH_SIDE, stride = TRAIN_STRIDE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
test_dataset = ChangeDetectionTestDataset(PATH_TO_DATASET, train = False, patch_side= PATCH_SIDE, stride = TRAIN_STRIDE)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 0)

print('DATASETS OK')

from torchvision import models



# 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands


if TYPE == 0:
    # net, net_name = Unet(3, N_FEATURES), 'FC-EF'
#     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
#      net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
       #net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
    net, net_name=FresUNet(2*3, N_FEATURES), "Strategy_4"
    #net, net_name=SiamUnet_diff_noDropout(3, N_FEATURES), 'FC-Siam-Diff'
net.apply(init_weights)

net.cuda()


true_rate =train_dataset.get_trueRate()

pos_weights_CD = torch.tensor([true_rate["shadow_CD"], true_rate["transparency_CD"], true_rate["texture_CD"]],
                           dtype=torch.float32).view(3, 1, 1).cuda()

pos_weights_LCM = torch.tensor([true_rate["shadow_LCM"], true_rate["transparency_LCM"], true_rate["texture_LCM"]],
                           dtype=torch.float32).view(3, 1, 1).cuda()


#pos_weights_CD=pos_weights_CD/pos_weights_CD.sum()
print(pos_weights_CD)
#pos_weights_LCM=pos_weights_LCM/pos_weights_LCM.sum()
print(pos_weights_LCM)


criterion_CD = nn.BCEWithLogitsLoss(pos_weight=pos_weights_CD[0])  
criterion_LCM = nn.BCEWithLogitsLoss(pos_weight=pos_weights_LCM[0])  
#criterion = nn.BCEWithLogitsLoss()  


print('NETWORK OK')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable parameters CD:', count_parameters(net))

# net.load_state_dict(torch.load('net-best_epoch-1_fm-0.7394933126157746.pth.tar'))

def train(n_epochs = N_EPOCHS, save = True):
    t = np.linspace(1, n_epochs, n_epochs)
    
    
    # Per la rete CD
    epoch_train_loss_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]

    epoch_train_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_dir_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_tex_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_precision_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_recall_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_Fmeasure_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]

    epoch_val_loss_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_dir_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_tex_accuracy_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_precision_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_recall_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_Fmeasure_CD = [[0] * N_FEATURES for _ in range(N_EPOCHS)]

    # Per la rete LCM
    epoch_train_loss_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_dir_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_tex_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_precision_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_recall_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_train_Fmeasure_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]

    epoch_val_loss_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_dir_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_tex_accuracy_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_precision_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_recall_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]
    epoch_val_Fmeasure_LCM = [[0] * N_FEATURES for _ in range(N_EPOCHS)]

   

    
    
    
#     mean_acc = 0
#     best_mean_acc = 0
  
    fm_CD = 0
    best_fm_CD = 0
    
    lss_CD = 1000
    best_lss_CD = 1000    
    
    
    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)
    
    
    optimizer = torch.optim.Adam([
    {'params': [param for name, param in net.named_parameters() if 'bias' in name], 'lr': 1e-2},
    {'params': [param for name, param in net.named_parameters() if 'bias' not in name], 'lr': 5e-2}], weight_decay=1e-4)
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    scaler=GradScaler()
    
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.90)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)

    
    for epoch_index in (range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))
        

#         for batch_index, batch in enumerate(tqdm(data_loader)):
        for batch in tqdm(train_loader):
            
            
            I1 = (batch['I1'].float().cuda()) #Reference
            I2 = (batch['I2'].float().cuda()) #Render
            imgs=[I1, I2]
           
            cm = ((batch['change map'].cuda()))
            cm=cm[:,0,:,:]
            cm=cm.unsqueeze(1)
            seg_ref=batch['segmentation reference'].cuda()
            seg_ref=seg_ref[:,0,:,:]
            seg_ref=seg_ref.unsqueeze(1)
            
            seg_ren=batch['segmentation render'].cuda()
            seg_ren=seg_ren[:,0,:,:]
            seg_ren=seg_ren.unsqueeze(1)
            

            optimizer.zero_grad() 
               
            output = net(I1, I2)

            loss_CD=criterion_CD(output[0], cm)
            loss_LCM=criterion_LCM(output[1], seg_ref)+criterion_LCM(output[2], seg_ren)
            
            #print(loss_CD+loss_LCM)
            scaler.scale(loss_CD+ALPHA*loss_LCM).backward()         
            scaler.step(optimizer)
            scaler.update()

            
        scheduler.step()

        epoch_train_loss_CD[epoch_index], epoch_train_accuracy_CD[epoch_index], cl_acc_CD, pr_rec_CD,epoch_train_loss_LCM[epoch_index], epoch_train_accuracy_LCM[epoch_index], cl_acc_LCM, pr_rec_LCM = validation(train_dataset)

        # Class accuracy e precision-recall per CD
        epoch_train_dir_accuracy_CD[epoch_index] = cl_acc_CD[0]
        #epoch_train_tex_accuracy_CD[epoch_index] = cl_acc_CD[1]
        epoch_train_precision_CD[epoch_index] = pr_rec_CD[0]
        epoch_train_recall_CD[epoch_index] = pr_rec_CD[1]
        epoch_train_Fmeasure_CD[epoch_index] = pr_rec_CD[2]

        # Class accuracy e precision-recall per LCM
        epoch_train_dir_accuracy_LCM[epoch_index] = cl_acc_LCM[0]
        #epoch_train_tex_accuracy_LCM[epoch_index] = cl_acc_LCM[1]
        epoch_train_precision_LCM[epoch_index] = pr_rec_LCM[0]
        epoch_train_recall_LCM[epoch_index] = pr_rec_LCM[1]
        epoch_train_Fmeasure_LCM[epoch_index] = pr_rec_LCM[2]

        # Unica chiamata alla funzione validation per il dataset di validazione
        epoch_val_loss_CD[epoch_index], epoch_val_accuracy_CD[epoch_index], cl_acc_CD, pr_rec_CD, epoch_val_loss_LCM[epoch_index], epoch_val_accuracy_LCM[epoch_index], cl_acc_LCM, pr_rec_LCM = validation(val_dataset)

        # Validation accuracy e precision-recall per CD
        epoch_val_dir_accuracy_CD[epoch_index] = cl_acc_CD[0]
        #epoch_val_tex_accuracy_CD[epoch_index] = cl_acc_CD[1]
        epoch_val_precision_CD[epoch_index] = pr_rec_CD[0]
        epoch_val_recall_CD[epoch_index] = pr_rec_CD[1]
        epoch_val_Fmeasure_CD[epoch_index] = pr_rec_CD[2]

        # Validation accuracy e precision-recall per LCM
        epoch_val_dir_accuracy_LCM[epoch_index] = cl_acc_LCM[0]
       # epoch_val_tex_accuracy_LCM[epoch_index] = cl_acc_LCM[1]
        epoch_val_precision_LCM[epoch_index] = pr_rec_LCM[0]
        epoch_val_recall_LCM[epoch_index] = pr_rec_LCM[1]
        epoch_val_Fmeasure_LCM[epoch_index] = pr_rec_LCM[2]

        # Plotting CD network metrics
        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss_CD[:epoch_index + 1], label='Train loss CD')
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_val_loss_CD[:epoch_index + 1], label='Validation loss CD')
        plt.legend(handles=[l1_1, l1_2], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.tight_layout()
        plt.title('Loss - CD')

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy_CD[:epoch_index + 1], label='Train accuracy CD')
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_val_accuracy_CD[:epoch_index + 1], label='Validation accuracy CD')
        plt.legend(handles=[l2_1, l2_2], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0, 100)
        plt.title('Accuracy - CD')
        plt.tight_layout()
        display.clear_output(wait=True)
        display.display(plt.gcf())

        # Plotting LCM network metrics
        plt.figure(num=4)
        plt.clf()
        l3_1, = plt.plot(t[:epoch_index + 1], epoch_train_loss_LCM[:epoch_index + 1], label='Train loss LCM')
        l3_2, = plt.plot(t[:epoch_index + 1], epoch_val_loss_LCM[:epoch_index + 1], label='Validation loss LCM')
        plt.legend(handles=[l3_1, l3_2], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.tight_layout()
        plt.title('Loss - LCM')

        plt.figure(num=5)
        plt.clf()
        l4_1, = plt.plot(t[:epoch_index + 1], epoch_train_accuracy_LCM[:epoch_index + 1], label='Train accuracy LCM')
        l4_2, = plt.plot(t[:epoch_index + 1], epoch_val_accuracy_LCM[:epoch_index + 1], label='Validation accuracy LCM')
        plt.legend(handles=[l4_1, l4_2], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0, 100)
        plt.title('Accuracy - LCM')
        plt.tight_layout()
        display.clear_output(wait=True)
        display.display(plt.gcf())

        # Plot Precision, Recall, and F-measure for CD
        plt.figure(num=3)
        plt.clf()
        l5_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision_CD[:epoch_index + 1], label='Train precision CD')
        l5_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall_CD[:epoch_index + 1], label='Train recall CD')
        l5_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure_CD[:epoch_index + 1], label='Train Dice/F1 CD')
        l5_4, = plt.plot(t[:epoch_index + 1], epoch_val_precision_CD[:epoch_index + 1], label='Validation precision CD')
        l5_5, = plt.plot(t[:epoch_index + 1], epoch_val_recall_CD[:epoch_index + 1], label='Validation recall CD')
        l5_6, = plt.plot(t[:epoch_index + 1], epoch_val_Fmeasure_CD[:epoch_index + 1], label='Validation Dice/F1 CD')
        plt.legend(handles=[l5_1, l5_2, l5_3, l5_4, l5_5, l5_6], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0, 1)
        plt.title('Precision, Recall, F-measure - CD')
        plt.tight_layout()
        display.clear_output(wait=True)
        display.display(plt.gcf())

        # Plot Precision, Recall, and F-measure for LCM
        plt.figure(num=6)
        plt.clf()
        l6_1, = plt.plot(t[:epoch_index + 1], epoch_train_precision_LCM[:epoch_index + 1], label='Train precision LCM')
        l6_2, = plt.plot(t[:epoch_index + 1], epoch_train_recall_LCM[:epoch_index + 1], label='Train recall LCM')
        l6_3, = plt.plot(t[:epoch_index + 1], epoch_train_Fmeasure_LCM[:epoch_index + 1], label='Train Dice/F1 LCM')
        l6_4, = plt.plot(t[:epoch_index + 1], epoch_val_precision_LCM[:epoch_index + 1], label='Validation precision LCM')
        l6_5, = plt.plot(t[:epoch_index + 1], epoch_val_recall_LCM[:epoch_index + 1], label='Validation recall LCM')
        l6_6, = plt.plot(t[:epoch_index + 1], epoch_val_Fmeasure_LCM[:epoch_index + 1], label='Validation Dice/F1 LCM')
        plt.legend(handles=[l6_1, l6_2, l6_3, l6_4, l6_5, l6_6], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0, 1)
        plt.title('Precision, Recall, F-measure - LCM')
        plt.tight_layout()
        display.clear_output(wait=True)
        display.display(plt.gcf())

        
        
           # Salvataggio del modello basato su F-measure per CD e LCM
        fm_CD = epoch_train_Fmeasure_CD[epoch_index]
        fm_LCM = epoch_train_Fmeasure_LCM[epoch_index]

        if fm_CD+fm_LCM > best_fm_CD:
            best_fm_CD = fm_CD+fm_LCM
            save_str_CD = 'net-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(fm_CD) + '.pth.tar'
            torch.save(net.state_dict(), save_str_CD)

      
        # Salvataggio del modello basato su loss per CD e LCM
        lss_CD = epoch_train_loss_CD[epoch_index]
        lss_LCM = epoch_train_loss_LCM[epoch_index]

        if lss_CD+lss_LCM < best_lss_CD:
            best_lss_CD = lss_CD+lss_LCM
            save_str_CD = 'net-best_epoch-' + str(epoch_index + 1) + '_loss-' + str(lss_CD) + '.pth.tar'
            torch.save(net.state_dict(), save_str_CD)

        
        # Salvataggio dei grafici
        if save:
            im_format = 'png'
            # Plot CD
            plt.figure(num=1)
            plt.savefig("Strategy4-CD" + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig("Strategy4-CD" + '-02-accuracy.' + im_format)

            plt.figure(num=3)
            plt.savefig("Strategy4-CD" + '-04-prec-rec-fmeas.' + im_format)

            # Plot LCM
            plt.figure(num=4)
            plt.savefig("Strategy4-LCM" + '-01-loss.' + im_format)

            plt.figure(num=5)
            plt.savefig("Strategy4-LCM" + '-02-accuracy.' + im_format)

            plt.figure(num=6)
            plt.savefig("Strategy4-LCM" + '-04-prec-rec-fmeas.' + im_format)

    # Output delle metriche
    out = {
        'train_loss_CD': epoch_train_loss_CD[-1],
        'train_accuracy_CD': epoch_train_accuracy_CD[-1],
        'train_dir_accuracy_CD': epoch_train_dir_accuracy_CD[-1],
        'train_tex_accuracy_CD': epoch_train_tex_accuracy_CD[-1],
        'test_loss_CD': epoch_val_loss_CD[-1],
        'test_accuracy_CD': epoch_val_accuracy_CD[-1],
        'test_dir_accuracy_CD': epoch_val_dir_accuracy_CD[-1],
        'test_tex_accuracy_CD': epoch_val_tex_accuracy_CD[-1],

        'train_loss_LCM': epoch_train_loss_LCM[-1],
        'train_accuracy_LCM': epoch_train_accuracy_LCM[-1],
        'train_dir_accuracy_LCM': epoch_train_dir_accuracy_LCM[-1],
        'train_tex_accuracy_LCM': epoch_train_tex_accuracy_LCM[-1],
        'test_loss_LCM': epoch_val_loss_LCM[-1],
        'test_accuracy_LCM': epoch_val_accuracy_LCM[-1],
        'test_dir_accuracy_LCM': epoch_val_dir_accuracy_LCM[-1],
        'test_tex_accuracy_LCM': epoch_val_tex_accuracy_LCM[-1]
    }

    print('pr_c_CD, rec_c_CD, f_meas_CD, pr_c_LCM, rec_c_LCM, f_meas_LCM')
    print(f'CD: {pr_rec_CD}, LCM: {pr_rec_LCM}')

    return out


N = 1

def validation(dset):

    net.eval()
    
    tot_loss_CD = 0
    tot_loss_LCM = 0
    tot_count = torch.tensor(0, dtype=torch.int32, device='cuda')
    
    
    n = N_FEATURES
    class_correct_CD = list(0. for i in range(n))
    class_total_CD = list(0. for i in range(n))
    class_accuracy_CD = list(0. for i in range(n)) 
    
    class_correct_LCM_ref = list(0. for i in range(n))
    class_total_LCM_ref = list(0. for i in range(n))
    class_accuracy_LCM_ref = list(0. for i in range(n))
    
    class_correct_LCM_ren= list(0. for i in range(n))
    class_total_LCM_ren = list(0. for i in range(n))
    class_accuracy_LCM_ren = list(0. for i in range(n))
    class_accuracy_LCM= list(0. for i in range(n))
    
    tp_CD = 0
    tn_CD = 0
    fp_CD = 0
    fn_CD = 0

    tp_LCM=0
    tn_LCM = 0
    fp_LCM = 0
    fn_LCM = 0


    
    for img_index in dset.names:
        for p in pov:
            
            I1_full, I2_full,cm_full, seg_ref, seg_ren = dset.get_img(img_index+"_"+p)
            
            
            I1 = (torch.unsqueeze(I1_full, 0).float()).cuda()
            I2 = (torch.unsqueeze(I2_full, 0).float()).cuda()
        
            cm = (torch.unsqueeze(cm_full, 0).float()).cuda()
            seg_ref = (torch.unsqueeze(seg_ref, 0).float()).cuda()
            seg_ren = (torch.unsqueeze(seg_ren, 0).float()).cuda()

            cm=cm[:,0,:,:]
            seg_ref=seg_ref[:,0,:,:]
            seg_ren=seg_ren[:,0,:,:]

            cm=cm.unsqueeze(1)
            seg_ren=seg_ren.unsqueeze(1)
            seg_ref=seg_ref.unsqueeze(1)
     
            prod=np.prod(cm.size())

            output= net(I1, I2)
            loss_CD=criterion_CD(output[0], cm)
            loss_LCM=criterion_LCM(output[1], seg_ref)+criterion_LCM(output[2], seg_ren)
            
                   
            tot_loss_LCM+=loss_LCM.data*prod 
            
            tot_loss_CD += loss_CD.data * prod 
            tot_count += prod

            output_sigmoid = tuple(torch.sigmoid(o) for o in output)
            output=output_sigmoid
           

            predicted_CD = (output[0].data ).int() 
            predicted_LCM_ref = (output[1].data ).int() 
            predicted_LCM_ren = (output[2].data ).int() 




            c_CD = (predicted_CD == cm.int())  
            c_LCM_ref = (predicted_LCM_ref == seg_ref.int())
            c_LCM_ren = (predicted_LCM_ren == seg_ren.int())
            
            for l in range(n):
                mask_CD = (cm[:, l, :, :] == 1)  # Maschera booleana dove cm == classe l
                class_correct_CD[l] = c_CD[:, l, :, :][mask_CD].sum()  # Corretto per la classe l
                class_total_CD[l] = mask_CD.sum()  # Conteggio totale della classe l
                
                mask_LCM_ref = (seg_ref[:, l, :, :] == 1)  # Maschera booleana dove cm == classe l
                class_correct_LCM_ref[l] = c_LCM_ref[:, l, :, :][mask_LCM_ref].sum()  # Corretto per la classe l
                class_total_LCM_ref[l] = mask_LCM_ref.sum()  # Conteggio totale della classe l
                
                mask_LCM_ren = (seg_ren[:, l, :, :] == 1)  # Maschera booleana dove cm == classe l
                class_correct_LCM_ren[l] = c_LCM_ren[:, l, :, :][mask_LCM_ren].sum()  # Corretto per la classe l
                class_total_LCM_ren[l] = mask_LCM_ren.sum()  # Conteggio totale della classe l
                    
            pr_CD = (predicted_CD > 0).cpu().numpy()
            gt_CD = (cm.int() > 0).cpu().numpy()
            
            tp_CD += np.logical_and(pr_CD, gt_CD).sum()  # True Positive: predizioni corrette
            tn_CD += np.logical_and(np.logical_not(pr_CD), np.logical_not(gt_CD)).sum()  # True Negative
            fp_CD += np.logical_and(pr_CD, np.logical_not(gt_CD)).sum()  # False Positive
            fn_CD += np.logical_and(np.logical_not(pr_CD), gt_CD).sum() 

            pr_LCM_ref = (predicted_LCM_ref > 0).cpu().numpy()
            gt_LCM_ref = (seg_ref.int() > 0).cpu().numpy()
            pr_LCM_ren = (predicted_LCM_ren > 0).cpu().numpy()
            gt_LCM_ren= (seg_ren.int() > 0).cpu().numpy()
            
            tp_LCM += (np.logical_and(pr_LCM_ref, gt_LCM_ref).sum()+np.logical_and(pr_LCM_ren, gt_LCM_ren).sum())/2  # True Positive: predizioni corrette
            tn_LCM += (np.logical_and(np.logical_not(pr_LCM_ref), np.logical_not(gt_LCM_ref)).sum()+ np.logical_and(np.logical_not(pr_LCM_ren), np.logical_not(gt_LCM_ren)).sum())/2  # True Negative
            fp_LCM += (np.logical_and(pr_LCM_ref, np.logical_not(gt_LCM_ref)).sum()+np.logical_and(pr_LCM_ren, np.logical_not(gt_LCM_ren)).sum())/2  # False Positive
            fn_LCM += (np.logical_and(np.logical_not(pr_LCM_ref), gt_LCM_ref).sum() + np.logical_and(np.logical_not(pr_LCM_ren), gt_LCM_ren).sum())/2

            
          
    net_loss_CD = ((tot_loss_CD)/tot_count).cpu()
    net_accuracy_CD = ((tp_CD + tn_CD)/tot_count).cpu()
    
    net_loss_LCM = ((tot_loss_LCM)/tot_count).cpu()
    net_accuracy_LCM = ((tp_LCM + tn_LCM)/(tot_count)).cpu()

    print("loss CD: ", net_loss_CD)
    print("loss LCM: ", net_loss_LCM)
    
    for i in range(n):
        class_accuracy_CD[i] = (100 * class_correct_CD[i] / max(class_total_CD[i],0.00001)).cpu()
        class_accuracy_LCM[i] = 100 * ((class_correct_LCM_ref[i] / max(class_total_LCM_ref[i],0.00001)).cpu()+(class_correct_LCM_ren[i] / max(class_total_LCM_ren[i],0.00001)).cpu())/2
        

    prec_CD = tp_CD / (tp_CD + fp_CD) if (tp_CD+fp_CD)!=0 else 0
    rec_CD = tp_CD / (tp_CD + fn_CD) if (tp_CD+fn_CD)!=0 else 0
    f_meas_CD = 2 * prec_CD * rec_CD / (prec_CD + rec_CD) if (prec_CD+rec_CD)!=0 else 0
    
    
    pr_rec_CD = [prec_CD, rec_CD, f_meas_CD]
    
    prec_LCM = tp_LCM / (tp_LCM + fp_LCM) if (tp_LCM+fp_LCM)!=0 else 0
    rec_LCM = tp_LCM / (tp_LCM + fn_LCM) if (tp_LCM+fn_LCM)!=0 else 0
    f_meas_LCM = 2 * prec_LCM * rec_LCM / (prec_LCM + rec_LCM) if (prec_LCM+rec_LCM)!=0 else 0
    
    
    pr_rec_LCM = [prec_LCM, rec_LCM, f_meas_LCM]
    print("precision, recall, F ",pr_rec_CD)
    
      
    return net_loss_CD, 100*net_accuracy_CD, class_accuracy_CD, pr_rec_CD, net_loss_LCM, 100*net_accuracy_LCM, class_accuracy_LCM, pr_rec_LCM
   
if LOAD_TRAINED:
    #net.load_state_dict(torch.load('net-best_epoch-31_fm-0.8676620879063464.pth.tar'))
    net.load_state_dict(torch.load('net_final.pth.tar'))
    print('LOAD OK')
else:
    t_start = time.time()
    out_dic = train()
    t_end = time.time()
    print(out_dic)
    print('Elapsed time:')
    print(t_end - t_start)


if not LOAD_TRAINED:
    torch.save(net.state_dict(), 'net_final.pth.tar')
    print('SAVE OK')


pred=0
def save_val_results(dset):
    net.eval()
    for name in tqdm(dset.names):
        for point_of_view in pov:
           
            with warnings.catch_warnings():
                I1, I2 = dset.get_img(name + "_" + point_of_view)

                # Converti l'immagine in tensore e spostalo su CUDA (GPU)
                I1 = (torch.unsqueeze(I1, 0).float()).cuda()
                I2 = (torch.unsqueeze(I2, 0).float()).cuda()

                # Definisci una matrice di segmentazione con shape (altezza, larghezza, 4)
                cm = np.zeros(shape=(img_height, img_width, N_FEATURES), dtype=np.uint8)
                seg_ref = np.zeros(shape=(img_height, img_width, N_FEATURES), dtype=np.uint8)
                seg_ren = np.zeros(shape=(img_height, img_width, N_FEATURES), dtype=np.uint8)

                output= net(I1, I2)

                output_CD=torch.sigmoid(output[0])
                output_LCM_ref=torch.sigmoid(output[1])
                output_LCM_ren=torch.sigmoid(output[2])


        
                predicted_CD=(output_CD).int()
                predicted_LCM_ref=(output_LCM_ref).int()
                predicted_LCM_ren=(output_LCM_ren).int()

                predicted_CD=predicted_CD.cpu().numpy()
                predicted_LCM_ref=predicted_LCM_ref.cpu().numpy()
                predicted_LCM_ren=predicted_LCM_ren.cpu().numpy()
                # Riempie le diverse classi (shadows, reflections, etc.) nei rispettivi canali

                for i in range(N_FEATURES): 
                    cm[:, :, i] = predicted_CD[:,i, :, :]   # shadows
                    seg_ref[:, :, i] = predicted_LCM_ref[:,i, :, :]   # shadows
                    seg_ren[:, :, i] = predicted_LCM_ren[:,i, :, :]   # shadows
                
                

                cm=cm.astype(np.uint8)

                for i in range(N_FEATURES):   
                    io.imsave(f'{net_name}-{name + "_" + point_of_view+"_"+class_index[i]}.png', cm[:, :, i]*255)
                    io.imsave(f'{net_name}-{name + "_" + point_of_view+"_"+class_index[i]}_REFERENCE.png', seg_ref[:, :, i]*255)
                    io.imsave(f'{net_name}-{name + "_" + point_of_view+"_"+class_index[i]}_RENDER.png', seg_ren[:, :, i]*255)
                

            
            
                

t_start = time.time()
# save_val_results(train_dataset)
pred=save_val_results(test_dataset)
t_end = time.time()
print('Elapsed time: {}'.format(t_end - t_start))



    
