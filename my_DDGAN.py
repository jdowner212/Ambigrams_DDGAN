import sys
import os
from os.path import dirname, abspath, join, exists

PARENT_DIR  = dirname(abspath(__file__))
if PARENT_DIR not in sys.path:
    sys.path = [PARENT_DIR] + sys.path

from general_utils import *

class Generator(torch.nn.Module):
    def __init__(self, node_factor=64, channels=1, img_size = 112, activation=Sigmoid):
        super().__init__()
        self.node_factor=node_factor
        self.activation = activation

        k_init = int(img_size/(2**3))

        self.gen = Sequential(
          ConvTranspose2d(in_channels = 100, out_channels = 4*node_factor, kernel_size = k_init, stride = 1, padding = 0, bias = False, device='cuda'),
          BatchNorm2d(num_features = 4*node_factor),
          LeakyReLU(0.2, inplace = True),

          ConvTranspose2d(in_channels = 4*node_factor, out_channels = 2*node_factor, kernel_size = 4, stride = 2, padding = 1, bias = False, device='cuda'),
          BatchNorm2d(num_features = 2*node_factor),
          LeakyReLU(0.2, inplace = True),

          ConvTranspose2d(in_channels = 2*node_factor, out_channels = node_factor, kernel_size = 4, stride = 2, padding = 1, bias = False, device='cuda'),
          BatchNorm2d(num_features = node_factor),
          LeakyReLU(0.2, inplace = True),

          ConvTranspose2d(in_channels = node_factor, out_channels = channels, kernel_size = 4, stride = 2, padding = 1, bias = False, device='cuda'),
          self.activation()
        )
    def forward(self, input):
        return self.gen(input).cuda()


class Discriminator(torch.nn.Module):
    def __init__(self, node_factor=64, image_channels=1, activation=Sigmoid):
        super().__init__()
        self.node_factor = node_factor
        self.activation = activation

        self.dis = Sequential(
            Conv2d(in_channels = image_channels, out_channels = self.node_factor, kernel_size = 2, stride = 2, padding = 1, bias=False, device='cuda'),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = self.node_factor, out_channels = self.node_factor*2, kernel_size = 4, stride = 2, padding = 1, bias=False, device='cuda'),
            BatchNorm2d(self.node_factor * 2),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = self.node_factor*2, out_channels = self.node_factor*4, kernel_size = 4, stride = 2, padding = 1, bias=False, device='cuda'),
            BatchNorm2d(self.node_factor * 4),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = self.node_factor*4, out_channels = self.node_factor*8, kernel_size = 4, stride = 2, padding = 1, bias=False, device='cuda'),
            BatchNorm2d(self.node_factor * 8),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = self.node_factor*8, out_channels = 1, kernel_size = 1, stride = 8, padding = 0, bias=False, device='cuda'),
            self.activation()
        )
    def forward(self, input):
        return self.dis(input)

def update_my_DDGAN(up_batch, down_batch, netD1, netD2, netG, opt_D1, opt_D2, opt_G, device): #, criterion):
    up_batch, down_batch = up_batch.cuda(), down_batch.cuda()

    ##################################
    # train D1 and D2
    ##################################

    opt_D1.zero_grad()
    opt_D2.zero_grad()
    
    #### 'REAL' examples ####
    
    # goal: train netD1 to recognize L1 images as 'real'
    output = netD1(up_batch)[:,0,0,0]
    target = torch.ones(len(up_batch), dtype=torch.float, device=device)
    lossD1_real = BCELoss()(output, target)
    lossD1_real.backward()
    
    # goal: train netD2 to recognize L2 images as 'real'
    output = netD2(down_batch)[:,0,0,0]
    target = torch.ones(len(down_batch), dtype=torch.float, device=device)
    lossD2_real = BCELoss()(output, target)
    lossD2_real.backward()
    

    #### 'FAKE' examples ####

    assert len(up_batch) == len(down_batch)
    noise = torch.randn(len(up_batch), 100, 1,1, device = device)
    fake_img = netG(noise) 

    # goal: train netD1 to recognize generated image as 'fake'
    output = netD1(fake_img.detach())[:,0,0,0]
    target = torch.zeros(len(up_batch), dtype=torch.float, device=device)
    lossD1_fake = BCELoss()(output, target)
    lossD1_fake.backward()
    
    # goal: train netD2 to recognize generated image as 'fake'
    output = netD2(fake_img.detach())[:,0,0,0]
    target = torch.zeros(len(down_batch), dtype=torch.float, device=device)
    lossD2_fake = BCELoss()(output, target)
    lossD2_fake.backward()
    
    opt_D1.step()
    opt_D2.step()
    
    ##################################
    # train G
    ##################################

    opt_G.zero_grad()

    output = netD1(fake_img)[:,0,0,0]
    # goal: make netG good enough to trick netD1 into thinking fake image is real
    target = torch.ones(len(up_batch), dtype=torch.float, device=device)
    lossG1 = BCELoss()(output, target)


    output = netD2(fake_img)[:,0,0,0]
    # goal: make netG good enough to trick netD1 into thinking fake image is real
    target = torch.ones(len(down_batch), dtype=torch.float, device=device)
    lossG2 = BCELoss()(output, target)

    
    lossG = lossG1 + lossG2
    lossG.backward()
    opt_G.step()

    return (netD1, netD2, netG), (opt_D1, opt_D2, opt_G), (lossD1_real, lossD2_real, lossD1_fake, lossD2_fake, lossG), fake_img
