from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import re

import random
# Imports

# PyTorch
from turtle import pos
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
# Models
from segment_anything import sam_model_registry, SamPredictor
from torchgeo.models import ChangeStar, ChangeMixin
# Other
import numpy as np
from skimage import io
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from math import  ceil

import time


print('IMPORTS OK')

# Global Variables' Definitions

N_FEATURES=4

THRESHOLD=0.5

PATH_TO_DATASET = 'Training/'
num_views=5
num_renders=20

BATCH_SIZE = 16
PATCH_SIDE = 512
N_EPOCHS = 100
img_height, img_width=512, 512
sleep_time=5


TRAIN_STRIDE = int(PATCH_SIDE /2) - 1


TYPE = 0 # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

LOAD_TRAINED = False

DATA_AUG = True

class_index=["Shadow","Reflections","Transparency","Texture"]
images_names_train=[]
images_names_val=[]
images_names_test=[]



labels_train=[ "donuts", "lego", "cuffie","bottiglia", 
"vino","strumenti","televisione", "scarpe", "palla",
"umanoide", "libro",  "caffe","martello", "crema", "scatole"]
labels_test=["forbici", "suzanne"]
labels_val=[ "geometrie", "ombrello", "carte"]



print('DEFINITIONS OK')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

print(device)

from skimage import transform
def read_sentinel_img(path):
    img = io.imread(path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    #img = img[:, :, ::-1]  # Inverte i canali: BGR -> RGB

    # Conversione a float32
    img = img.astype('float32')

    # Normalizzazione (se necessario)
    if img.max() > 1.0:
        img /= 255.0

    return img
def read_sentinel_img_trio_reference(path, obj, nw):
    I1 = read_sentinel_img(path+str(nw)+"_0000_"+obj+"_Reference_Image.png") #REFERENCE
    return I1

def read_sentinel_img_trio_cm(path, obj, nw, nr):
  
    nr_str = f"{nr:04d}"
    I2 = read_sentinel_img(path+str(nw)+"_"+nr_str+"_"+obj+"_Render_Image.png") #Render

    cm_shadow = io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_CMShadow.png", as_gray=True)
    cm_shadow=(cm_shadow!=0).astype(np.float32)

    cm_transparency=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_CMTransparency.png", as_gray=True)
    cm_transparency=(cm_transparency!=0).astype(np.float32)
    
    cm_texture=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_CMTexture.png", as_gray=True)
    cm_texture=(cm_texture!=0).astype(np.float32)
    
    cm_reflections=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_CMReflections.png", as_gray=True)
    cm_reflections=(cm_reflections!=0).astype(np.float32)
    cm=np.stack((cm_shadow, cm_reflections,  cm_transparency, cm_texture), axis=2)
    cm = np.transpose(cm, (2, 0, 1))    
    cm = torch.from_numpy(cm).float() 

    return  I2, cm

def read_sentinel_img_trio_seg(path, obj, nw, nr, ref_ren):

    nr_str = f"{nr:04d}"
  
    seg_shadow = io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_"+ref_ren+"_Shadow.png", as_gray=True)
    seg_shadow=(seg_shadow/seg_shadow.max()).astype(np.float32) if seg_shadow.max()!=0 else seg_shadow
    seg_shadow=(seg_shadow!=0).astype(np.float32)

    seg_transparency=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_"+ref_ren+"_Transparency.png", as_gray=True)
    seg_transparency=(seg_transparency/seg_transparency.max()).astype(np.float32) if seg_transparency.max()!=0 else seg_transparency
    seg_transparency=(seg_transparency!=0).astype(np.float32)
    
    seg_texture=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_"+ref_ren+"_Texture.png", as_gray=True)   
    seg_texture=(seg_texture/seg_texture.max()).astype(np.float32) if seg_texture.max()!=0 else seg_texture
    seg_texture=(seg_texture!=0).astype(np.float32)
    
    seg_reflections=io.imread(path+str(nw)+"_"+nr_str+"_"+obj+"_"+ref_ren+"_Reflections.png", as_gray=True)
    seg_reflections=(seg_reflections/seg_reflections.max()).astype(np.float32) if seg_reflections.max()!=0 else seg_reflections
    seg_reflections=(seg_reflections!=0).astype(np.float32)
    seg=np.stack((seg_shadow,seg_reflections,  seg_transparency, seg_texture), axis=2)
    seg = np.transpose(seg, (2, 0, 1)) 
   
    seg = torch.from_numpy(seg).float() 

    return seg

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
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
        if train=="Test":
            self.names = labels_test 
        
        

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
            for nw  in range(num_views):
                I1=read_sentinel_img_trio_reference(self.path + obj_img+"/", obj_img, nw)
                name_label=f"{nw}_0000_{obj_img}"
                self.imgs_1[name_label] = reshape_for_torch(I1)
                seg_reference=read_sentinel_img_trio_seg(self.path + obj_img+"/", obj_img, nw, nr=0, ref_ren="Reference")
                self.seg_reference[name_label] = seg_reference

                true_pix_shadow_LCM += (seg_reference[0].sum()) #0 ombre, 1 trasparenze, 2 texture
                true_pix_reflections_LCM +=(seg_reference[1].sum())
                true_pix_transparency_LCM +=(seg_reference[2].sum())
                true_pix_texture_LCM +=(seg_reference[3].sum())
                

                for nr in range(1, num_renders):
              
                    # load and store each image
                    I2, cm = read_sentinel_img_trio_cm(self.path + obj_img+"/", obj_img, nw, nr) #entra nella cartella dell'oggetto
                    seg_render=read_sentinel_img_trio_seg(self.path + obj_img+"/", obj_img,nw, nr, ref_ren="Render")
                    name_label=f"{nw}_{nr:04}_{obj_img}"
                    self.imgs_2[name_label] = reshape_for_torch(I2)
                    
                    self.seg_render[name_label] = seg_render

                    if(train==True):
                        images_names_train.append(name_label)
                    if(train==False):
                        images_names_val.append(name_label)
                    if(train=="Test"):
                        images_names_test.append(name_label)
                    
                    
                    self.change_map[name_label] = cm
                    
                    s = cm.shape
                    n_pix += np.prod(s)
                    true_pix_shadow_CM += cm[0].sum() #0 ombre, 1 trasparenze, 2 texture
                    true_pix_reflections_CM += cm[1].sum()
                    true_pix_transparency_CM +=cm[2].sum()
                    true_pix_texture_CM +=cm[3].sum()
                    
                    true_pix_shadow_LCM += (seg_render[0].sum()) #0 ombre, 1 trasparenze, 2 texture
                    true_pix_reflections_LCM +=(seg_render[1].sum())
                    true_pix_transparency_LCM +=(seg_render[2].sum())
                    true_pix_texture_LCM +=(seg_render[3].sum())
                    
                    
                    # calculate the number of patches
                    s = self.imgs_2[name_label].shape
                    n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
                    n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
                    n_patches_i = n1 * n2
                    self.n_patches_per_image[name_label] = n_patches_i
                    self.n_patches += n_patches_i
                    
                    # generate path coordinates
                    for i in range(n1):
                        for j in range(n2):
                            # coordinates in (x1, x2, y1, y2)
                            current_patch_coords = (name_label, 
                                            [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                            [self.stride*(i + 1), self.stride*(j + 1)])
                            self.patch_coords.append(current_patch_coords)
        
                   

        n_pix=n_pix/4
        self.shadow_rate_CM=(n_pix-true_pix_shadow_CM)/true_pix_shadow_CM if true_pix_shadow_CM!=0 else 0
        self.texture_rate_CM=(n_pix-true_pix_texture_CM)/true_pix_texture_CM if true_pix_texture_CM!=0 else 0
        self.transparency_rate_CM=(n_pix-true_pix_transparency_CM)/true_pix_transparency_CM if true_pix_transparency_CM!=0 else 0
        self.reflection_rate_CM=(n_pix-true_pix_reflections_CM)/true_pix_reflections_CM if true_pix_reflections_CM!=0 else 0
        
        self.shadow_rate_LCM=(2*n_pix-true_pix_shadow_LCM)/true_pix_shadow_LCM if true_pix_shadow_LCM!=0 else 0
        self.texture_rate_LCM=(2*n_pix-true_pix_texture_LCM)/true_pix_texture_LCM if true_pix_texture_LCM!=0 else 0
        self.transparency_rate_LCM=(2*n_pix-true_pix_transparency_LCM)/true_pix_transparency_LCM if true_pix_transparency_LCM!=0 else 0
        self.reflections_rate_LCM=(2*n_pix-true_pix_reflections_LCM)/true_pix_reflections_LCM if true_pix_reflections_LCM!=0 else 0
    
    def get_trueRate(self):
        return {'shadow_CD':self.shadow_rate_CM, 'reflections_CD': self.reflection_rate_CM,
                "transparency_CD": self.transparency_rate_CM, "texture_CD": self.texture_rate_CM, 
                'reflections_LCM': self.reflections_rate_LCM,
                "shadow_LCM": self.shadow_rate_LCM, "transparency_LCM": self.transparency_rate_LCM, "texture_LCM": self.texture_rate_LCM}

    def get_img(self, im_name):
        im_name_ref=re.sub(r'_(\d+)_', r'_0000_', im_name)
        return self.imgs_1[im_name_ref], self.imgs_2[im_name],self.change_map[im_name], self.seg_reference[im_name_ref], self.seg_render[im_name]
    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        im_name_ref=re.sub(r'_(\d+)_', r'_0000_', im_name)
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]     
        I1 = self.imgs_1[im_name_ref][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        label = self.change_map[im_name][:,limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()
        segmentation_reference=self.seg_reference[im_name_ref][:,limits[0]:limits[1], limits[2]:limits[3]]
        segmentation_reference = torch.from_numpy(1*np.array(segmentation_reference)).float()        
        segmentation_render=self.seg_render[im_name][:,limits[0]:limits[1], limits[2]:limits[3]]
        segmentation_render = torch.from_numpy(1*np.array(segmentation_render)).float()
        sample = {'I1': I1, "I2": I2, 'change map': label, 'segmentation reference': segmentation_reference, 'segmentation render': segmentation_render}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
class RandomHorizontalFlip(object):
    """Flip randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        I1,I2, label, seg_ref, seg_ren = sample['I1'], sample['I2'],sample['change map'], sample['segmentation reference'], sample['segmentation render']
        
        
        if random.random() > 0.5:
            # Flip I1 and I2
            I1 = torch.flip(I1, dims=[2])  # Flip sull'ultima dimensione (orizzontale)
            I2 = torch.flip(I2, dims=[2])
            
            # Flip label, seg_ref, seg_ren
            label = torch.flip(label, dims=[2])
            seg_ref = torch.flip(seg_ref, dims=[2])
            seg_ren = torch.flip(seg_ren, dims=[2])
            
           
            
        return {
            'I1': I1, 
            'I2': I2,  
            'change map': label, 
            'segmentation reference': seg_ref, 
            'segmentation render': seg_ren
        }
class RandomVerticalFlip(object):
    """Flip the images in a sample vertically with a random probability."""

    def __call__(self, sample):
        I1, I2, label, seg_ref, seg_ren = sample['I1'], sample['I2'], sample['change map'], sample['segmentation reference'], sample['segmentation render']
       
        
        # Perform vertical flipping with a probability (e.g., 90% chance)
        if random.random() > 0.5:
            # Flip delle immagini
            I1 = torch.flip(I1, dims=[1])  # Flip sull'asse verticale (altezza)
            I2 = torch.flip(I2, dims=[1])

            # Flip della change map e delle mappe di segmentazione
            label = torch.flip(label, dims=[1])
            seg_ref = torch.flip(seg_ref, dims=[1])
            seg_ren = torch.flip(seg_ren, dims=[1])

        # Restituzione del campione aggiornato
        return {
            'I1': I1,
            'I2': I2,
            'change map': label,
            'segmentation reference': seg_ref,
            'segmentation render': seg_ren
        }

class RandomRot(object):


    def __call__(self, sample):
        I1,I2, label, seg_ref, seg_ren = sample['I1'], sample['I2'],sample['change map'], sample['segmentation reference'], sample['segmentation render']

     
        
        n = random.randint(0, 3)  # Numero di rotazioni di 90 gradi (0, 1, 2, 3)

        if n > 0:  # Se n == 0, non applicare nessuna rotazione
            # Ruota immagini RGB
            I1 = torch.rot90(I1, k=n, dims=[1, 2])  # Ruota sugli assi [H, W]
            I2 = torch.rot90(I2, k=n, dims=[1, 2])

            # Ruota le mappe
            label = torch.rot90(label, k=n, dims=[-2, -1])  # Ruota sugli assi [H, W]
            seg_ref = torch.rot90(seg_ref, k=n, dims=[-2, -1])
            seg_ren = torch.rot90(seg_ren, k=n, dims=[-2, -1])

        # Restituisce il dizionario con le immagini ruotate
        return {
            'I1': I1,
            'I2': I2,
            'change map': label,
            'segmentation reference': seg_ref,
            'segmentation render': seg_ren
        }
    
print('UTILS OK')
data_aug=None
if DATA_AUG:
    data_aug = transforms.Compose([
    RandomHorizontalFlip()])

train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = True, stride = TRAIN_STRIDE,patch_side=PATCH_SIDE, transform=data_aug)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = False, patch_side= PATCH_SIDE, stride = TRAIN_STRIDE, transform=None)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train = "Test", patch_side= PATCH_SIDE, stride = TRAIN_STRIDE, transform=None)
#test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 0)
print('DATASETS OK')
sam = sam_model_registry["vit_b"](checkpoint="segment-anything\sam_vit_b_01ec64.pth")

for param in sam.parameters():
    param.requires_grad = False

# Aggiungi un livello di output personalizzato
class ShadowSAM(nn.Module):
    def __init__(self, sam):
        super(ShadowSAM, self).__init__()
        self.sam = sam
        '''self.new_output = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, N_FEATURES, kernel_size=5, padding=2))'''
        self.new_output=nn.Conv2d(256, N_FEATURES, kernel_size=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        original_size = x.shape[-2:]  # Memorizza le dimensioni originali [H, W]
        output_list = []
        predictor = SamPredictor(self.sam)
        # Itera su ogni immagine nel batch
        for i in range(batch_size):
            image = x[i]  # Estrai la singola immagine
            image = image.permute(1, 2, 0) 
            predictor.set_image(image.cpu().numpy())  # Setta l'immagine singola (convertita in numpy)
            features = predictor.features.to(x.device)  # Ottieni le feature e porta a device
            mask = self.new_output(features)

            mask_resized = F.interpolate(mask, size=original_size, mode="bilinear", align_corners=False)

            output_list.append((mask_resized))
     
        return torch.stack(output_list)

class DenseFeatureExtractor(nn.Module):
    def __init__(self, shadow_sam):
        super(DenseFeatureExtractor, self).__init__()
        self.sam = shadow_sam.sam  # Usa SAM come estrattore di caratteristiche, senza il layer finale

    def forward(self, x):
        batch_size = x.shape[0]
        original_size = x.shape[-2:]
        features_list = []
        predictor = SamPredictor(self.sam)
        for i in range(batch_size):
            image = x[i].permute(1, 2, 0)
            predictor.set_image(image.cpu().numpy())
            features = predictor.features.to(x.device)
            features_list.append(features)
  
        features_tensor = torch.stack(features_list, dim=0)  
    
        return features_tensor

# Classificatore per la segmentazione
class SegmentationClassifier(nn.Module):
    def __init__(self, shadow_sam):
        super(SegmentationClassifier, self).__init__()
        self.classifier = shadow_sam.new_output  # Usa l'ultimo livello per generare maschere binarie
    
        
    def forward(self, features):
        output_list = []
        mask= self.classifier(features)    
        mask_resized = F.interpolate(mask, size=img_height, mode="bilinear", align_corners=False)
        output_list.append((mask_resized))
        
        # Ricomponi il batch dalle singole maschere
        return torch.stack(output_list)





# Configurazione di ChangeMixin
changemixin = ChangeMixin(size=img_height, out_channels=N_FEATURES)

# Inizializza il modello ShadowSAM
seg_net, seg_net_name = ShadowSAM(sam), "SAM"
#seg_net.load_state_dict(torch.load("shadow_sam_model.pth.tar"))

dense_feature_extractor = DenseFeatureExtractor(seg_net)
seg_classifier = SegmentationClassifier(seg_net)
net, net_name = ChangeStar(
    dense_feature_extractor=dense_feature_extractor,
    seg_classifier=seg_classifier,
    changemixin=changemixin,
    inference_mode="t1t2"  # Modalità di inferenza
), "ChangeStar"

net.load_state_dict(torch.load("ChangeStar_model_final.pth.tar"))
net.to(device)


true_rate =train_dataset.get_trueRate()

print(true_rate)

weights_CD = torch.tensor([true_rate["shadow_CD"], true_rate["reflections_CD"], true_rate["transparency_CD"], true_rate["texture_CD"]],
                           dtype=torch.float32).view(4, 1, 1).cuda()

weights_LCM = torch.tensor([true_rate["shadow_LCM"],true_rate["reflections_LCM"], true_rate["transparency_LCM"], true_rate["texture_LCM"]],
                           dtype=torch.float32).view(4, 1, 1).cuda()

weights_CD=weights_CD/weights_CD.sum()
weights_LCM=weights_LCM/weights_LCM.sum()
print(weights_CD)
print(weights_LCM)
criterion_CD = nn.BCEWithLogitsLoss(weight=weights_CD)
criterion_LCM = nn.BCEWithLogitsLoss(weight=weights_LCM)

print('NETWORK OK')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(y_pred, y_true, chunk_size=10_000_000):
    

    tp = tn = fp = fn = 0

    # Itera sui chunk
    for i in range(0, len(y_true), chunk_size):
        # Estrai il chunk corrente
        y_true_chunk = y_true[i:i + chunk_size]
        y_pred_chunk = y_pred[i:i + chunk_size]
        

        # Calcola TP, TN, FP, FN per questo chunk
        tp += np.logical_and(y_pred_chunk, y_true_chunk).sum()
        tn += np.logical_and(np.logical_not(y_pred_chunk), np.logical_not(y_true_chunk)).sum()
        fp += np.logical_and(y_pred_chunk, np.logical_not(y_true_chunk)).sum()
        fn += np.logical_and(np.logical_not(y_pred_chunk), y_true_chunk).sum()

    # Ora calcoliamo le metriche finali su tutti i dati
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return precision, recall, f1, iou, accuracy

def validation():
    net.eval()
    val_loss = 0
    y_true_CD, y_pred_CD = [], []
    y_true_LCM_ref, y_pred_LCM_ref = [], []
    y_true_LCM_ren, y_pred_LCM_ren = [], []
    
    with torch.no_grad():

        for batch_val in tqdm(val_loader):
           
            I1 = (batch_val['I1'].to(device)) #Reference
            I2 = (batch_val['I2'].to(device)) #Render
            
            I1=(I1*255).byte()
            I2=(I2*255).byte()        
            cm = ((batch_val['change map'].float().to(device)))
            seg_ref=batch_val['segmentation reference'].float().to(device)
            seg_ren=batch_val['segmentation render'].float().to(device)
            #seg_ren=seg_ren[:,0,:,:]      
            #seg_ref=seg_ref[:,0,:,:]  
            #cm=cm[:,0,:,:]    
            #cm=cm.unsqueeze(1)
            #seg_ref=seg_ref.unsqueeze(1)
            #seg_ren=seg_ren.unsqueeze(1)
            outputs = net(I1, I2)
            out_ref, out_ren=outputs["bi_seg_logit"][0], outputs["bi_seg_logit"][1]           
            out_map=outputs['change_prob']           
                        
            loss_seg = (criterion_LCM(out_ref,seg_ref )+criterion_LCM(out_ren, seg_ren))*0.5
            loss_map=criterion_CD(out_map, cm)
            loss=loss_map+loss_seg
            
            val_loss+=loss.item()
            out_ref = torch.sigmoid(out_ref)
            out_ren = torch.sigmoid(out_ren)
            out_map = torch.sigmoid(out_map)

            y_true_CD.append((cm > THRESHOLD).cpu().numpy().flatten())
            y_pred_CD.append((out_map > THRESHOLD).cpu().numpy().flatten())
            y_true_LCM_ref.append((seg_ref > THRESHOLD).cpu().numpy().flatten())
            y_true_LCM_ren.append((seg_ren > THRESHOLD).cpu().numpy().flatten())
            y_pred_LCM_ref.append((out_ref > THRESHOLD).cpu().numpy().flatten())
            y_pred_LCM_ren.append((out_ren > THRESHOLD).cpu().numpy().flatten())
            
            
            torch.cuda.empty_cache()
            time.sleep(sleep_time)
           
            
            
        y_true_CD = np.concatenate(y_true_CD)
        y_pred_CD = np.concatenate(y_pred_CD)
        y_true_LCM_ref = np.concatenate(y_true_LCM_ref)
        y_pred_LCM_ref = np.concatenate(y_pred_LCM_ref)
        y_true_LCM_ren = np.concatenate(y_true_LCM_ren)
        y_pred_LCM_ren = np.concatenate(y_pred_LCM_ren)
        
        precision_CD, recall_CD, f1_CD, iou_CD, accuracy_CD = compute_metrics(y_true=y_true_CD, y_pred=y_pred_CD)
        precision_LCM_ref, recall_LCM_ref, f1_LCM_ref, iou_LCM_ref, accuracy_LCM_ref = compute_metrics(y_true=y_true_LCM_ref, y_pred=y_pred_LCM_ref)
        precision_LCM_ren, recall_LCM_ren, f1_LCM_ren, iou_LCM_ren, accuracy_LCM_ren = compute_metrics(y_true=y_true_LCM_ren, y_pred=y_pred_LCM_ren)
        
        precision_LCM=(precision_LCM_ref+precision_LCM_ren)/2
        recall_LCM=(recall_LCM_ref+recall_LCM_ren)/2
        f1_LCM=(f1_LCM_ref+f1_LCM_ren)/2
        iou_LCM=(iou_LCM_ref+iou_LCM_ren)/2
        accuracy_LCM=(accuracy_LCM_ref+accuracy_LCM_ren)/2 

    return val_loss, precision_CD, recall_CD, f1_CD, iou_CD, accuracy_CD, precision_LCM, recall_LCM, f1_LCM, iou_LCM, accuracy_LCM


print('Number of trainable parameters:', count_parameters(net))
scaler = torch.amp.GradScaler(device="cuda")

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)
def visualizza_immagine(tensor, titolo, cmap=None):
    """
    Funzione per visualizzare un tensore come immagine.
    """
    if len(tensor.shape) == 4:  # Batch con canali
        tensor = tensor[0]  # Prendiamo il primo elemento del batch
    if tensor.shape[0] == 3:  # 3 Canali (RGB)
        tensor = tensor.permute(1, 2, 0)  # Trasporre i canali (C, H, W) -> (H, W, C)
    elif tensor.shape[0] == 1:  # Canale singolo
        tensor = tensor.squeeze(0)  # Rimuoviamo il canale
    img = tensor.cpu().numpy()  # Convertiamo in numpy
    plt.imshow(img, cmap=cmap)
    plt.title(titolo)
    plt.axis('off')
if LOAD_TRAINED==False:
    min_loss=100
    train_loss= [[0] for _ in range(N_EPOCHS)]
    val_loss= [[0] for _ in range(N_EPOCHS)]
    metrics = {
    "val_CD": ["accuracy", "precision", "recall", "f1", "iou"],
    "val_LCM": ["accuracy", "precision", "recall", "f1", "iou"],
    "train_CD": ["accuracy", "precision", "recall", "f1", "iou"],
    "train_LCM": ["accuracy", "precision", "recall", "f1", "iou"]
    }

    # Initialize metric lists
    metric_values = {key: {metric: [] for metric in metrics[key]} for key in metrics}

    # Initialize true and predicted values
    
    t = np.linspace(1, N_EPOCHS, N_EPOCHS)
    for epoch in range(N_EPOCHS):
        net.train()
        epoch_loss = 0
        y_true_CD, y_pred_CD = [], []
        y_true_LCM_ref, y_pred_LCM_ref = [], []
        y_true_LCM_ren, y_pred_LCM_ren = [], []
        for batch in tqdm(train_loader):
            
            I1 = (batch['I1'].to(device)) #Reference
            I2 = (batch['I2'].to(device)) #Render
            
            I1=(I1*255).byte()
            I2=(I2*255).byte()
        
            cm = ((batch['change map'].float().to(device)))
            #cm=cm[:,0,:,:]
            #cm=cm.unsqueeze(1)
            seg_ref=batch['segmentation reference'].float().to(device)
            #seg_ref=seg_ref[:,0,:,:]
            #seg_ref=seg_ref.unsqueeze(1)
            
            seg_ren=batch['segmentation render'].float().to(device)
            #seg_ren=seg_ren[:,0,:,:]
            #seg_ren=seg_ren.unsqueeze(1)

            '''plt.figure(figsize=(12, 8))

            # I1
            plt.subplot(2, 3, 1)
            visualizza_immagine(I1, "I1 (Reference)")

            # I2
            plt.subplot(2, 3, 2)
            visualizza_immagine(I2, "I2 (Render)")

            # cm
            plt.subplot(2, 3, 3)
            visualizza_immagine(cm, "Change Map", cmap='gray')

            # seg_ref
            plt.subplot(2, 3, 4)
            visualizza_immagine(seg_ref, "Segmentation Reference", cmap='gray')

            # seg_ren
            plt.subplot(2, 3, 5)
            visualizza_immagine(seg_ren, "Segmentation Render", cmap='gray')

            plt.tight_layout()
            plt.show()'''

            # Inizializza il gradiente
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                # Esegui la predizione
                outputs = net(I1, I2)
                out_ref, out_ren=outputs["bi_seg_logit"][0], outputs["bi_seg_logit"][1]
                out_map=outputs['bi_change_logit']

                out_map=out_map[:,0,:,:,:]
                #out_ref=out_ref.float()
                #out_ren=out_ren.float()
                
                loss_seg = (criterion_LCM(out_ref,seg_ref )+criterion_LCM(out_ren, seg_ren))*0.5
                loss_map=criterion_CD(out_map, cm)
            loss=loss_map+loss_seg
            
            epoch_loss += loss.item()
            scaler.scale(loss).backward()  # Scala il gradiente
            scaler.step(optimizer)
            scaler.update()
            out_ref = torch.sigmoid(out_ref)
            out_ren = torch.sigmoid(out_ren)
            out_map = torch.sigmoid(out_map)


            y_true_CD.append((cm > THRESHOLD).cpu().numpy().flatten())
            y_pred_CD.append((out_map > THRESHOLD).cpu().numpy().flatten())
            y_true_LCM_ref.append((seg_ref > THRESHOLD).cpu().numpy().flatten())
            y_true_LCM_ren.append((seg_ren > THRESHOLD).cpu().numpy().flatten())
            y_pred_LCM_ref.append((out_ref > THRESHOLD).cpu().numpy().flatten())
            y_pred_LCM_ren.append((out_ren > THRESHOLD).cpu().numpy().flatten())
            
            
            torch.cuda.empty_cache()
            time.sleep(sleep_time)
           
            
            
        y_true_CD = np.concatenate(y_true_CD)
        y_pred_CD = np.concatenate(y_pred_CD)
        y_true_LCM_ref = np.concatenate(y_true_LCM_ref)
        y_pred_LCM_ref = np.concatenate(y_pred_LCM_ref)
        y_true_LCM_ren = np.concatenate(y_true_LCM_ren)
        y_pred_LCM_ren = np.concatenate(y_pred_LCM_ren)
        
        precision_CD, recall_CD, f1_CD, iou_CD, accuracy_CD = compute_metrics(y_true=y_true_CD, y_pred=y_pred_CD)
        precision_LCM_ref, recall_LCM_ref, f1_LCM_ref, iou_LCM_ref, accuracy_LCM_ref = compute_metrics(y_true=y_true_LCM_ref, y_pred=y_pred_LCM_ref)
        precision_LCM_ren, recall_LCM_ren, f1_LCM_ren, iou_LCM_ren, accuracy_LCM_ren = compute_metrics(y_true=y_true_LCM_ren, y_pred=y_pred_LCM_ren)
        precision_LCM=(precision_LCM_ref+precision_LCM_ren)/2
        recall_LCM=(recall_LCM_ref+recall_LCM_ren)/2
        f1_LCM=(f1_LCM_ref+f1_LCM_ren)/2
        iou_LCM=(iou_LCM_ref+iou_LCM_ren)/2
        accuracy_LCM=(accuracy_LCM_ref+accuracy_LCM_ren)/2

        train_loss[epoch]=epoch_loss/len(train_loader)
        metric_values["train_CD"]["accuracy"].append(accuracy_CD*100)
        metric_values["train_CD"]["precision"].append(precision_CD)
        metric_values["train_CD"]["recall"].append(recall_CD)
        metric_values["train_CD"]["f1"].append(f1_CD)
        metric_values["train_CD"]["iou"].append(iou_CD)

        metric_values["train_LCM"]["accuracy"].append(accuracy_LCM*100)
        metric_values["train_LCM"]["precision"].append(precision_LCM)
        metric_values["train_LCM"]["recall"].append(recall_LCM)
        metric_values["train_LCM"]["f1"].append(f1_LCM)
        metric_values["train_LCM"]["iou"].append(iou_LCM)

        val_loss[epoch], val_precision_CD, val_recall_CD, val_f1_CD, val_iou_CD, val_accuracy_CD,val_precision_LCM, val_recall_LCM, val_f1_LCM, val_iou_LCM, val_accuracy_LCM= validation()
    

        metric_values["val_CD"]["accuracy"].append(val_accuracy_CD*100)
        metric_values["val_CD"]["precision"].append(val_precision_CD)
        metric_values["val_CD"]["recall"].append(val_recall_CD)
        metric_values["val_CD"]["f1"].append(val_f1_CD)
        metric_values["val_CD"]["iou"].append(val_iou_CD)

        metric_values["val_LCM"]["accuracy"].append(val_accuracy_LCM*100)
        metric_values["val_LCM"]["precision"].append(val_precision_LCM)
        metric_values["val_LCM"]["recall"].append(val_recall_LCM)
        metric_values["val_LCM"]["f1"].append(val_f1_LCM)
        metric_values["val_LCM"]["iou"].append(val_iou_LCM)

        print(f"Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Validation Loss: {val_loss[epoch]:.4f}")
        print(metric_values)

        plt.figure(num=1)
        plt.clf()
        l1_1, = plt.plot(t[:epoch + 1], train_loss[:epoch+ 1], label='Train loss')
        l1_2, = plt.plot(t[:epoch + 1], val_loss[:epoch+ 1], label='Validation loss')
        plt.legend(handles=[l1_1, l1_2], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.tight_layout()
        plt.title('Loss')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.figure(num=2)
        plt.clf()
        l2_1, = plt.plot(t[:epoch + 1], metric_values["val_LCM"]["precision"][:epoch+ 1], label='Validation Precision')
        l2_2, = plt.plot(t[:epoch + 1], metric_values["val_LCM"]["recall"][:epoch+ 1], label='Validation Recall')
        l2_3, = plt.plot(t[:epoch + 1], metric_values["val_LCM"]["f1"][:epoch+ 1], label='Validation F1-score')
        l2_4, = plt.plot(t[:epoch + 1], metric_values["val_LCM"]["iou"][:epoch+ 1], label='Validation IoU')
        l2_5, = plt.plot(t[:epoch + 1], metric_values["train_LCM"]["precision"][:epoch+ 1], label='Train Precision')
        l2_6, = plt.plot(t[:epoch + 1], metric_values["train_LCM"]["recall"][:epoch+ 1], label='Train Recall')
        l2_7, = plt.plot(t[:epoch + 1], metric_values["train_LCM"]["f1"][:epoch+ 1], label='Train F1-score')
        l2_8, = plt.plot(t[:epoch + 1], metric_values["train_LCM"]["iou"][:epoch+ 1], label='Train IoU')
        plt.legend(handles=[l2_1, l2_2, l2_3, l2_4, l2_5, l2_6,l2_7,l2_8], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.title('Metrics - LCM')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.figure(num=3)
        plt.clf()
        l3_1, = plt.plot(t[:epoch + 1], metric_values["val_LCM"]["accuracy"][:epoch+ 1], label='Validation Accuracy - LCM')
        l3_2, = plt.plot(t[:epoch + 1], metric_values["train_LCM"]["accuracy"][:epoch+ 1], label='Train Accuracy - LCM')
        l3_3, = plt.plot(t[:epoch + 1], metric_values["val_CD"]["accuracy"][:epoch+ 1], label='Validation Accuracy - CD')
        l3_4, = plt.plot(t[:epoch + 1], metric_values["train_CD"]["accuracy"][:epoch+ 1], label='Train Accuracy - CD')
        plt.legend(handles=[l3_1, l3_2, l3_3, l3_4], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0,100)
        plt.tight_layout()
        plt.title('Accuracy')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.figure(num=4)
        plt.clf()
        l4_1, = plt.plot(t[:epoch + 1], metric_values["val_CD"]["precision"][:epoch+ 1], label='Validation Precision')
        l4_2, = plt.plot(t[:epoch + 1], metric_values["val_CD"]["recall"][:epoch+ 1], label='Validation Recall')
        l4_3, = plt.plot(t[:epoch + 1], metric_values["val_CD"]["f1"][:epoch+ 1], label='Validation F1-score')
        l4_4, = plt.plot(t[:epoch + 1], metric_values["val_CD"]["iou"][:epoch+ 1], label='Validation IoU')
        l4_5, = plt.plot(t[:epoch + 1], metric_values["train_CD"]["precision"][:epoch+ 1], label='Train Precision')
        l4_6, = plt.plot(t[:epoch + 1], metric_values["train_CD"]["recall"][:epoch+ 1], label='Train Recall')
        l4_7, = plt.plot(t[:epoch + 1], metric_values["train_CD"]["f1"][:epoch+ 1], label='Train F1-score')
        l4_8, = plt.plot(t[:epoch + 1], metric_values["train_CD"]["iou"][:epoch+ 1], label='Train IoU')
        plt.legend(handles=[l4_1, l4_2, l4_3, l4_4, l4_5, l4_6,l4_7,l4_8], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid()
        plt.gcf().gca().set_xlim(left=1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.title('Metrics - CD')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        im_format = 'png'
        # Plot CD
        plt.figure(num=1)
        plt.savefig("ChangeStar" + '-Loss.' + im_format)
        plt.figure(num=2)
        plt.savefig("ChangeStar" + '-Metrics-LCM.' + im_format)
        plt.figure(num=3)
        plt.savefig("ChangeStar" + '-Accuracy.' + im_format)
        plt.figure(num=4)
        plt.savefig("ChangeStar" + '-Metrics-CD.' + im_format)
        if((train_loss[epoch]+val_loss[epoch]<min_loss and epoch%2==0)):
            min_loss=train_loss[epoch]+val_loss[epoch]
            loss_str="ChangeStar_model.pth.tar"
            torch.save(net.state_dict(),loss_str)


    torch.save(net.state_dict(), "ChangeStar_model_final.pth.tar")

else:
    
    net.load_state_dict(torch.load("ChangeStar_model_final.pth.tar"))

net.eval()
with torch.no_grad():
    #for j, f in enumerate(class_index):
    for i in range(len(images_names_test)):
        
   
        
        I1, I2, _, _, _ =test_dataset.get_img(images_names_test[i])   
        
        I1=I1.unsqueeze(0)
        I2=I2.unsqueeze(0)
        I1=(I1*255).byte().to(device)
        I2=(I2*255).byte().to(device)

        
        output = net(I1,I2)

        
        
        
        out_ref, out_ren=output["bi_seg_logit"][0], output["bi_seg_logit"][1]
        out_map=output["change_prob"]
        out_ref = (out_ref).sigmoid().squeeze().cpu().numpy()
        out_ren = (out_ren).sigmoid().squeeze().cpu().numpy()
        out_map= (out_map).sigmoid().squeeze().cpu().numpy()
        out_ref = ((out_ref) * 255).astype(np.uint8)
        out_ren = ((out_ren) * 255).astype(np.uint8)
        out_map = ((out_map) * 255).astype(np.uint8)

        time.sleep(sleep_time)

        
        # Salva l'immagine direttamente senza una colormap
        for j, f in enumerate(class_index):
            plt.imsave(f'{net_name}-{images_names_test[i]}_{f}_CM.png', out_map[j,:,:], cmap='gray')
            plt.imsave(f'{net_name}-{images_names_test[i]}_{f}_Reference.png', out_ref[j,:,:], cmap='gray')
            plt.imsave(f'{net_name}-{images_names_test[i]}_{f}_Render.png', out_ren[j,:,:], cmap='gray')
            