import torch
import numpy as np
import os,random
import torchvision
from pro_gan_pytorch import  Encoder , Networks as net
#import tensorflow as tf
import skimage
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

set_seed(6)

netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
netD = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
netD.load_state_dict(torch.load('/_yucheng/bigModel/pro-gan/PGGAN/result/RC_1/models/D_model_ep0.pth',map_location=device))
#netD.load_state_dict(torch.load('../E-model/E/D_model_ep0.pth',map_location=device))
# netD2 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD2.load_state_dict(torch.load('../E-model/E/D_model_ep1.pth',map_location=device))
# netD3 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD3.load_state_dict(torch.load('../E-model/E/D_model_ep2.pth',map_location=device))
# netD4 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD4.load_state_dict(torch.load('../E-model/E/D_model_ep3.pth',map_location=device))
# netD5 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD5.load_state_dict(torch.load('../E-model/E/D_model_ep4.pth',map_location=device))
# netD6 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD6.load_state_dict(torch.load('../E-model/E/D_model_ep5.pth',map_location=device))
# netD7 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD7.load_state_dict(torch.load('../E-model/E/D_model_ep6.pth',map_location=device))
# netD8 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD8.load_state_dict(torch.load('../E-model/E/D_model_ep7.pth',map_location=device))
# netD9 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD9.load_state_dict(torch.load('../E-model/E/D_model_ep8.pth',map_location=device))
# netD10 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD10.load_state_dict(torch.load('../E-model/E/D_model_ep9.pth',map_location=device))

# #netD = Encoder.encoder_v2() #新结构，不需要参数 

z = torch.randn(8, 512).to(device)
with torch.no_grad():
	x = netG(z,depth=8,alpha=1)
	z_ = netD(x.detach(),height=8,alpha=1)
	z_ = z_.squeeze(2).squeeze(2)
	#z_ = netD(x.detach()) #new_small_Net , 或者注释前两行
	x_ = netG(z_,depth=8,alpha=1)


# #--------------------PSNR & SSIM------------------
psnr_all_1=0
#psnr_all_2=0
ssim_all_1=0
#ssim_all_2=0
for i in range(8):
	array1 = x[i].cpu().numpy().squeeze()
	array1 = array1.transpose(1,2,0)
	#array1 = (array1+1)/2
	array2 = x_[i].cpu().numpy().squeeze()
	array2 = array2.transpose(1,2,0)
	#array2 = (array2+1)/2
	psnr1 = skimage.measure.compare_psnr(array1, array2, 255)
	psnr_all_1 +=psnr1
	#psnr2 = tf.image.psnr(array1, array2, max_val=255)
	# print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
	# print(psnr1)
	# print('-------------')
	# print(psnr2)
	# print('-------------')
	ssim1 = skimage.measure.compare_ssim(array1, array2, data_range=255,multichannel=True)
	ssim_all_1 +=ssim1
	#ssim2 = tf.image.ssim(tf.convert_to_tensor(array1),tf.convert_to_tensor(array2),max_val=255)
	# print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
	# print(ssim1)
	# print('-------------')
	# print(ssim2)
	# print('-------------')

print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
print(psnr_all_1/8)
print('-------------')
# print(psnr_all_2/10)
# print('-------------')
print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
print(ssim_all_1/8)
print('-------------')
# print(ssim_all_2/10)
# print('-------------')

# #----------------show image---------
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(array1)
# plt.subplot(1,2,2)
# plt.imshow(array2)
# plt.show()
#a = matplotlib.image.imread('../v5_200.jpg')
#a = matplotlib.image.imread('../Fs_MNIST_Encoder/gan_samples_rc_v4/ep0_249.jpg')
#a = matplotlib.image.imread('../ep9_4000.jpg')
#array1 = a[:,3078:4105,:] #HQ:(2054, 8210, 3)
# array1 = a[:,1026:2053,:] #HQ:(2054, 8210, 3)
# matplotlib.image.imsave('../t2.png', array1)
# print(a.shape)
#array2 = a[67:,:,:]
#b = b[67:,:,:]

#-------------------LPIPS --- code-------------
#import sys
#sys.path.append('PerceptualSimilarity')
# from PerceptualSimilarity.util import util
# import PerceptualSimilarity.models as models
# from PerceptualSimilarity.models import dist_model as dm
# from IPython import embed


# use_gpu = False         # Whether to use GPU
# spatial = True         # Return a spatial map of perceptual distance.

# # Linearly calibrated models (LPIPS)
# model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True) #spatial
# dummy_im0 = x # image should be RGB, normalized to [-1,1]
# dummy_im1 = x_
# if(use_gpu):
# 	dummy_im0 = dummy_im0.cuda()
# 	dummy_im1 = dummy_im1.cuda()
# dist = model.forward(dummy_im0,dummy_im1)

# print('dist:'+str(dist))

import lpips
loss_fn_alex = lpips.LPIPS(net='alex',use_gpu=True) # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg',use_gpu=True)
img1= x.cuda(0)
img2 = x_.cuda(0)
d1 = loss_fn_alex(img1, img2)
d2 = loss_fn_vgg(img1, img2)
print('dist_alex:'+str(d1))
print('dist_vgg:'+str(d2))
# #----------------save image---------
array1 = (array1+1)/2
array2 = (array2+1)/2
matplotlib.image.imsave('./z_8.png', array1)
matplotlib.image.imsave('./A1_8.png', array2)
