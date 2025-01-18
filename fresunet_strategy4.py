# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2019. Multitask learning for large-scale semantic change detection. Computer Vision and Image Understanding, 187, p.102783.



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_ss(nn.Module):

    def __init__(self, inplanes, planes = None, subsamp=1):
        super(BasicBlock_ss, self).__init__()
        if planes == None:
            planes = inplanes * subsamp
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.subsamp = subsamp
        self.doit = planes != inplanes
        if self.doit:
            self.couple = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.doit:
            residual = self.couple(x)
            residual = self.bnc(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.subsamp > 1:
            out = F.max_pool2d(out, kernel_size=self.subsamp, stride=self.subsamp)
            residual = F.max_pool2d(residual, kernel_size=self.subsamp, stride=self.subsamp)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)

        return out
    

    
class BasicBlock_us(nn.Module):

    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp) # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1, bias=False) 
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=1, mode='bilinear', align_corners=False)
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        #out = F.interpolate(out, scale_factor=1, mode='bilinear', align_corners=False)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
class FresUNetEncoderCD(nn.Module):
    """Encoder part of FresUNet."""
    
    def __init__(self, input_nbr):
        super(FresUNetEncoderCD, self).__init__()
        
        cur_depth = input_nbr
        base_depth = 8
        
        # Encoding stage 1
        self.encres1_1 = BasicBlock_ss(cur_depth, planes=base_depth)
        cur_depth = base_depth
        self.encres1_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 2
        self.encres2_1 = BasicBlock_ss(cur_depth)
        self.encres2_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 3
        self.encres3_1 = BasicBlock_ss(cur_depth)
        self.encres3_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
        # Encoding stage 4
        self.encres4_1 = BasicBlock_ss(cur_depth)
        self.encres4_2 = BasicBlock_ss(cur_depth, subsamp=2)
        cur_depth *= 2
        
    def forward(self, x):
        # Encoding forward pass
        x1 = self.encres1_1(x)
        x = self.encres1_2(x1)
        
        x2 = self.encres2_1(x)
        x = self.encres2_2(x2)
        
        x3 = self.encres3_1(x)
        x = self.encres3_2(x3)
        
        x4 = self.encres4_1(x)
        x = self.encres4_2(x4)
        
        return x1, x2, x3, x4, x



class FresUNetDecoderCD(nn.Module):
    """Decoder part of FresUNet."""
    
    def __init__(self, base_depth, label_nbr):
        super(FresUNetDecoderCD, self).__init__()
        
        cur_depth = base_depth * 16  # Depth from the deepest part of the encoder

        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss(cur_depth)
        self.decres4_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss(cur_depth + base_depth * 8, planes=cur_depth)
        self.decres3_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss(cur_depth + base_depth * 4, planes=cur_depth)
        self.decres2_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss(cur_depth + base_depth * 2, planes=cur_depth)
        self.decres1_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Output
        self.coupling = nn.Conv2d(cur_depth + base_depth, label_nbr, kernel_size=1, bias=False)
       # self.sm = nn.Sigmoid()
        
    def forward(self, x, x1, x2, x3, x4):
       

        # Decoding forward pass
        x = self.decres4_1(x)
        x = self.decres4_2(x)
        
        x = self.decres3_1(torch.cat((x, x4), 1))
        x = self.decres3_2(x)
        
        x = self.decres2_1(torch.cat((x, x3), 1))
        x = self.decres2_2(x)
        
        x = self.decres1_1(torch.cat((x, x2), 1))
        x = self.decres1_2(x)
        
        x = self.coupling(torch.cat((x, x1), 1))
        #x = self.sm(x)
        
        return x
class FresUNetDecoderLCM(nn.Module):
    """Decoder part of FresUNet."""
    
    def __init__(self, base_depth, label_nbr):
        super(FresUNetDecoderLCM, self).__init__()
        
        cur_depth = base_depth * 16  # Depth from the deepest part of the encoder

        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss(cur_depth)
        self.decres4_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss(cur_depth + base_depth * 8, planes=cur_depth)
        self.decres3_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss(cur_depth + base_depth * 4, planes=cur_depth)
        self.decres2_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss(cur_depth + base_depth * 2, planes=cur_depth)
        self.decres1_2 = BasicBlock_us(cur_depth, upsamp=2)
        cur_depth = int(cur_depth / 2)
        
        # Output
        self.coupling = nn.Conv2d(cur_depth + base_depth, label_nbr, kernel_size=1, bias=False)
        #self.sm = nn.Sigmoid()
        
    def forward(self, x, x1, x2, x3, x4):
        # Decoding forward pass
        x = self.decres4_1(x)
        x = self.decres4_2(x)
        
        x = self.decres3_1(torch.cat((x, x4), 1))
        x = self.decres3_2(x)
        
        x = self.decres2_1(torch.cat((x, x3), 1))
        x = self.decres2_2(x)
        
        x = self.decres1_1(torch.cat((x, x2), 1))
        x = self.decres1_2(x)
        
        x = self.coupling(torch.cat((x, x1), 1))
        #x = self.sm(x)
        
        
        return x

    
class FresUNet(nn.Module):
    """FresUNet segmentation network."""
    
    def __init__(self, input_nbr, label_nbr):
        super(FresUNet, self).__init__()


        '''self.encoderCD = FresUNetEncoderCD(input_nbr)
        self.decoderCD = FresUNetDecoderCD(base_depth=8*3, label_nbr=label_nbr)
        self.encoderLCM_ref=FresUNetEncoderCD(input_nbr=int(input_nbr/2))
        self.encoderLCM_ren=FresUNetEncoderCD(input_nbr=int(input_nbr/2))
        self.decoderLCM_ref = FresUNetDecoderLCM(base_depth=8, label_nbr=label_nbr)
        self.decoderLCM_ren = FresUNetDecoderLCM(base_depth=8, label_nbr=label_nbr)'''

        self.encoderCD = FresUNetEncoderCD(input_nbr)
        self.decoderCD = FresUNetDecoderCD(base_depth=8*2, label_nbr=label_nbr)
        self.encoderLCM_ref=FresUNetEncoderCD(input_nbr=int(input_nbr))
        self.decoderLCM_ref = FresUNetDecoderLCM(base_depth=8, label_nbr=label_nbr*2)
        
    def forward(self, x1, x2):
        # Concatenation of the two inputs
        x = torch.cat((x1, x2), 1)
        
        
        # Forward pass through the encoder
        x1_CD, x2_CD, x3_CD, x4_CD, x_CD = self.encoderCD(x)
        
        #x1_LCM_ref, x2_LCM_ref, x3_LCM_ref, x4_LCM_ref, x_LCM_ref = self.encoderLCM_ref(x)
        x1_LCM, x2_LCM, x3_LCM, x4_LCM, x_LCM = self.encoderLCM_ref(x)
        
        #x1_LCM_ren, x2_LCM_ren, x3_LCM_ren, x4_LCM_ren, x_LCM_ren= self.encoderLCM_ren(x2)

        x_f=torch.cat((x_CD, x_LCM), 1)
        x1_f=torch.cat((x1_CD, x1_LCM), 1)
        x2_f=torch.cat((x2_CD, x2_LCM), 1)
        x3_f=torch.cat((x3_CD, x3_LCM), 1)
        x4_f=torch.cat((x4_CD, x4_LCM), 1)

        
        
        
        x_LCM = self.decoderLCM_ref(x_LCM, x1_LCM, x2_LCM, x3_LCM, x4_LCM)



        x_CD = self.decoderCD(x_f, x1_f, x2_f, x3_f, x4_f)

        
        x_LCM=torch.split(x_LCM, int(x_LCM.shape[1]/2),dim=1)
        return x_CD, x_LCM[0], x_LCM[1]
