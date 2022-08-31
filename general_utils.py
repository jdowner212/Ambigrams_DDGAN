import os
from os.path import dirname, abspath, join, exists
import sys

PARENT_DIR  = dirname(abspath(__file__))
DATA_DIR    = join(PARENT_DIR, 'data')
MODELS_DIR  = join(PARENT_DIR, 'models')
FONTS_DIR   = join(PARENT_DIR, 'fonts')
FONT_COUNT  = len([f for f in os.listdir(FONTS_DIR) if f[0] != '.'])

if PARENT_DIR not in sys.path:
    sys.path = [PARENT_DIR] + sys.path
 

import torch
from   torch import clamp, FloatTensor, optim, zeros_like, ones_like, stack, tensor
from   torch.nn import Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Tanh, Sigmoid, Flatten, BCELoss
from   torchvision.transforms import Compose, Normalize, RandomAffine, RandomRotation, RandomOrder, ToPILImage, ToTensor
from   itertools import product
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from   PIL import Image, ImageFont, ImageDraw
import random
import shutil
import time
from   tqdm.notebook import tqdm

# import Generator
# import Discriminator


rc            = random.choice
random_affine = RandomAffine  (degrees=10, fill=(255,255,255), shear=10)
random_rotate = RandomRotation(degrees=8,  fill=(255,255,255))
random_Ts     = RandomOrder([random_affine, random_rotate])
denorm        = Compose([Normalize(mean = [ 0.,      0.,      0.      ],
                                   std  = [ 1/0.229, 1/0.224, 1/0.225 ]),
                         Normalize(mean = [ -0.485,  -0.456,  -0.406  ], 
                                   std  = [ 1.,      1.,      1.      ])])


noise                         = lambda t: FloatTensor(t.shape).uniform_(-0.5, 0.5)
add_noise                     = lambda t: clamp( t+noise(t), zeros_like(t), ones_like(t) )
rot180                        = lambda x: torch.rot90 ( torch.rot90(x,1,[2,3]), 1, [2,3])
open_as_grayscale_tensor      = lambda filepath, size: ToTensor()(Image.open(open(filepath,'rb')).convert('L').resize((size,size)))
current_epoch                 = lambda path: int(path.split('/')[-3].split('_')[1]) + 1
largest_epoch                 = lambda path: max([int(f.split('_')[1]) for f in os.listdir(join(MODELS_DIR, model_name_from_path(path))) if 'epoch' in f]) + 1
model_name_from_path          = lambda path: path.split('/')[-4]
log_path_from_model_path      = lambda path: ('/'.join(path.split('/')[:-3]) + '/log.txt')


def val(original, random_delta=0):
    if str(type(original)) == "<class 'list'>":
        return rc(original)
    elif str(type(original)) != "<class 'list'>" and random_delta != 0:
        lower = max(0.00001, original-random_delta)
        upper = min(0.99999, original+random_delta)
        return round(random.uniform(lower,upper), 4)
    else:
        return original

def try_to_make_folder(path, print_message=None):
    '''
    create folder and handle exceptions.
    '''
    try:
        os.mkdir(path)
    except:
        if print_message:
            print(print_message)

def model_as_string(model_name, lr_D, lr_G, wghts_dec_D, wghts_dec_G, Bs_D, Bs_G, D1_act, G_act, nf, batch_sz):
    ret = f'Model {model_name}:\n\n\
           \n' + ' '*19 + f'lr_D = {lr_D},\
           \n' + ' '*19 + f'lr_G = {lr_G},\
           \n' + ' '*19 + f'weights_dec_D = {wghts_dec_D},\
           \n' + ' '*19 + f'weights_dec_D = {wghts_dec_G},\
           \n' + ' '*19 + f'betas_D = {Bs_D},\
           \n' + ' '*19 + f'betas_D = {Bs_G},\
           \n' + ' '*19 + f'netD_activation = {str(D1_act())[:-2]},\
           \n' + ' '*19 + f'netG_activation = {str(G_act())[:-2]},\
           \n' + ' '*19 + f'node_factor = {nf},\
           \n\nbatch_size = {batch_sz},'
    return ret