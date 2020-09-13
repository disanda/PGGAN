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
from skimage.io import imsave

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def set_seed(seed):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed) # cpu
#     torch.cuda.manual_seed_all(seed)  # gpu
#     torch.backends.cudnn.deterministic = True


# ep = 'ep0'
# netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
# netD = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# #netD = torch.nn.DataParallel(Encoder.encoder_v2())
# #netD.load_state_dict(torch.load('/_yucheng/bigModel/pro-gan/PGGAN/result/RC_3_new_samll_Net/models/D_model_'+ep+'.pth',map_location=device))
# #netD.load_state_dict(torch.load('../E-model/E/D_model_ep0.pth',map_location=device))
# # netD2 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD2.load_state_dict(torch.load('../E-model/E/D_model_ep1.pth',map_location=device))
# # netD3 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD3.load_state_dict(torch.load('../E-model/E/D_model_ep2.pth',map_location=device))
# # netD4 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD4.load_state_dict(torch.load('../E-model/E/D_model_ep3.pth',map_location=device))
# # netD5 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD5.load_state_dict(torch.load('../E-model/E/D_model_ep4.pth',map_location=device))
# # netD6 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD6.load_state_dict(torch.load('../E-model/E/D_model_ep5.pth',map_location=device))
# # netD7 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD7.load_state_dict(torch.load('../E-model/E/D_model_ep6.pth',map_location=device))
# # netD8 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD8.load_state_dict(torch.load('../E-model/E/D_model_ep7.pth',map_location=device))
# # netD9 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# # netD9.load_state_dict(torch.load('../E-model/E/D_model_ep8.pth',map_location=device))
# # netD10 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
# netD.load_state_dict(torch.load('../E-model/E/D_model_ep9.pth',map_location=device))

# #netD = Encoder.encoder_v2() #新结构，不需要参数 


#------------------随机数生成图像----------------
# set_seed(20)
# z = torch.randn(1, 512).to(device)
# with torch.no_grad():
# 	x = netG(z,depth=8,alpha=1)
# 	z_ = netD(x.detach(),height=8,alpha=1)
# 	z_ = z_.squeeze(2).squeeze(2)
# 	#z_ = netD(x.detach()) #new_small_Net , 或者注释前两行
# 	x_ = netG(z_,depth=8,alpha=1)
# #y = (torch.cat((x[:8],x_[:8]))+1)/2
# #torchvision.utils.save_image(y, '../fig_seed_6.png',nrow=2)
# torchvision.utils.save_image((x+1)/2, '../Gz_1.png')
# torchvision.utils.save_image((x_+1)/2, '../Gx_1.png')

# torchvision.utils.save_image(x, '../Gz_2.png',normalize=True)
# torchvision.utils.save_image(x_, '../Gx_2.png',normalize=True)



#----------------encoding & decoding------------
# from sklearn import preprocessing
# #img = matplotlib.image.imread('../1.jpg') ## (0,255)
# # matplotlib.image.imsave('../rc_1_real.png', img)

# img1 = Image.open('../1.jpg') # [w,h,c]  (0,255)
# #img = img.resize((64,64))
# img1 = np.array(img1) 
# # for i in range(3):
# # 	img1[i] = preprocessing.normalize(img1[i], norm='l1') # -->[-1,1]


# img1 = img1.transpose(2,0,1) #[w,h,c]->[c,w,h]
# print(img1.shape)
# img1 = torch.tensor(img1).type(torch.FloatTensor)
# img1 = img1.view((1,3,1024,1024))
# print(img1.shape)
# with torch.no_grad():
# 	z_ = netD(img1,height=8,alpha=1)
# 	z_ = z_.squeeze(2).squeeze(2)
# 	#z_ = netD(x.detach()) #new_small_Net , 或者注释前两行
# 	x_ = netG(z_,depth=8,alpha=1)
# torchvision.utils.save_image(x_,'../rc_1.png',normalize=True)
# # img=img.squeeze().numpy().transpose(1,2,0)
# # matplotlib.image.imsave('../rc_2_real.png', img)
# # #torchvision.utils.save_image((x_+1)/2,'../rc_0.png')


# resultPath = "./metrics/"
# if not os.path.exists(resultPath):
#     os.mkdir(resultPath)

# resultPath1_1 = resultPath+"/E_small"
# if not os.path.exists(resultPath1_1):
#     os.mkdir(resultPath1_1)

# resultPath1_1_1 = resultPath1_1+"/"+ep
# if not os.path.exists(resultPath1_1_1):
#     os.mkdir(resultPath1_1_1)

# resultPath1_1_2 = resultPath1_1+"/True_"+ep
# if not os.path.exists(resultPath1_1_2):
#     os.mkdir(resultPath1_1_2)

# #--------------------PSNR & SSIM------------------
# psnr_all_1=0
# #psnr_all_2=0
# ssim_all_1=0
# #ssim_all_2=0
# for i in range(16):
# 	array1 = x[i].cpu().numpy().squeeze()
# 	array1 = array1.transpose(1,2,0)
# 	#array1 = (array1+1)/2
# 	array2 = x_[i].cpu().numpy().squeeze()
# 	array2 = array2.transpose(1,2,0)
# 	#array2 = (array2+1)/2
# 	psnr1 = skimage.measure.compare_psnr(array1, array2, 255)
# 	psnr_all_1 +=psnr1
# 	#psnr2 = tf.image.psnr(array1, array2, max_val=255)
# 	# print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
# 	# print(psnr1)
# 	# print('-------------')
# 	# print(psnr2)
# 	# print('-------------')
# 	ssim1 = skimage.measure.compare_ssim(array1, array2, data_range=255,multichannel=True)
# 	ssim_all_1 +=ssim1
# 	#ssim2 = tf.image.ssim(tf.convert_to_tensor(array1),tf.convert_to_tensor(array2),max_val=255)
# 	# print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
# 	# print(ssim1)
# 	# print('-------------')
# 	# print(ssim2)
# 	# print('-------------')
# 	#img1 = (array1+1)/2
# 	#img2 = (array2+1)/2
# 	# matplotlib.image.imsave(resultPath1_1_1+'./rc_%d.png'%i, img1) #报错,应该是浮点数类型不对
# 	# matplotlib.image.imsave(resultPath1_1_2+'./Gz_%d.png'%i, img2)
# 	imsave(resultPath1_1_2+'/Gz_%d.png'%i, array1)
# 	imsave(resultPath1_1_1+'/rc_%d.png'%i, array2)
# 	print('doing:'+str(i))
# print('-------------') #PSNR的单位是dB，数值越大表示失真越小。20-40dB
# print(psnr_all_1/16)
# print('-------------')
# # print(psnr_all_2/10)
# # print('-------------')
# print('-------------') #SSIM取值范围[0,1]，值越大，表示图像失真越小.
# print(ssim_all_1/16)
# print('-------------')
# # print(ssim_all_2/10)
# # print('-------------')

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
#-----------------------LPIPS -----pip------------------
# import lpips
# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg')
# loss_fn_alex.cuda()
# loss_fn_vgg.cuda()
# img1= x[:10]
# img2 = x_[:10]
# d1 = loss_fn_alex(img1, img2)
# d2 = loss_fn_vgg(img1, img2)
# print('dist_alex:'+str(d1.mean()))
# print('dist_vgg:'+str(d2.mean()))

# # #----------------save image---------
# # array1 = (array1+1)/2
# # array2 = (array2+1)/2
# # matplotlib.image.imsave('./z_8.png', array1)
# # matplotlib.image.imsave('./A1_8.png', array2)
# #torchvision.utils.save_image(y, save_dir + '/%d_Epoch-c_c.png' % i)
# y = (torch.cat((x[:8],x_[:8]))+1)/2
# dir_img = '/E_'+ep
# torchvision.utils.save_image(y, resultPath1_1+dir_img+'.png',nrow=8)
# torchvision.utils.save_image((x[:8]+1)/2, resultPath1_1+dir_img+'_Gz.png',nrow=8)
# torchvision.utils.save_image((x_[:8]+1)/2, resultPath1_1+dir_img+'_rc.png',nrow=8)

img = matplotlib.image.imread('../fig_seed_5.png') ## (0,255)
print(img.shape)
img = img[:4105,:,:]
matplotlib.image.imsave('../aaaaaaa.png', img)



