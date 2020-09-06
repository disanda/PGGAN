import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import  Encoder , Networks as net
import tensorflow as tf
import skimage
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
# netD = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD.load_state_dict(torch.load('../E-model/E/D_model_ep1.pth',map_location=device))
# #netD = Encoder.encoder_v2() #新结构，不需要参数 

# z = torch.randn(1, 512).to(device)
# with torch.no_grad():
# 	x = netG(z,depth=8,alpha=1)
# 	z_ = netD(x.detach(),height=8,alpha=1)
# 	z_ = z_.squeeze(2).squeeze(2)
# 	#z_ = netD(x.detach()) #new_small_Net , 或者注释前两行
# 	x_ = netG(z_,depth=8,alpha=1)


# #--------------------PSNR & SSIM------------------

# array1 = x.numpy().squeeze()
# array1 = array1.transpose(1,2,0)
# array1 = (array1+1)/2
# array2 = x_.numpy().squeeze()
# array2 = array2.transpose(1,2,0)
# array2 = (array2+1)/2

# psnr1 = skimage.measure.compare_psnr(array1, array2, 255)
# psnr2 = tf.image.psnr(array1, array2, max_val=255)

# print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
# print(psnr1)
# print('-------------')
# print(psnr2)
# print('-------------')

# ssim1 = skimage.measure.compare_ssim(array1, array2, data_range=255,multichannel=True)
# ssim2 = tf.image.ssim(tf.convert_to_tensor(array1),tf.convert_to_tensor(array2),max_val=255)

# print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
# print(ssim1)
# print('-------------')
# print(ssim2)
# print('-------------')


# #----------------show image---------
# matplotlib.image.imsave('t1.png', array1)
# matplotlib.image.imsave('t2.png', array2)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(array1)
# plt.subplot(1,2,2)
# plt.imshow(array2)
# plt.show()


#a = matplotlib.image.imread('../v5_200.jpg')
#a = matplotlib.image.imread('../Fs_MNIST_Encoder/gan_samples_rc_v4/ep0_249.jpg')
a = matplotlib.image.imread('../ep9_4000.jpg')
#array1 = a[:,3078:4105,:] #HQ:(2054, 8210, 3)
array1 = a[:,1026:2053,:] #HQ:(2054, 8210, 3)
matplotlib.image.imsave('../t2.png', array1)
print(a.shape)
#array2 = a[67:,:,:]
#b = b[67:,:,:]

# psnr1 = skimage.measure.compare_psnr(array1, array2, 255)
# psnr2 = tf.image.psnr(array1, array2, max_val=255)

# print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
# print(psnr1)
# print('-------------')
# print(psnr2)
# print('-------------')

# ssim1 = skimage.measure.compare_ssim(array1, array2, data_range=255,multichannel=True)
# ssim2 = tf.image.ssim(tf.convert_to_tensor(array1),tf.convert_to_tensor(array2),max_val=255)

# print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
# print(ssim1)
# print('-------------')
# print(ssim2)
# print('-------------')


#-------------------LPIPS----------------
