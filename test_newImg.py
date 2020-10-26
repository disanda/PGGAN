import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import PRO_GAN , Encoder , Networks as net
from pro_gan_pytorch.DataTools import DatasetFromFolder

#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#-------------------model load-------------
netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 


netD = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
netD.load_state_dict(torch.load('./pre-model/AG_D_model_ep9.pth',map_location=device))



#------------------测试重构图像-----------
data_path='./newImage/'
#data_path='/Users/apple/Desktop/CelebAMask-HQ/CelebA-HQ-img'

trans = torchvision.transforms.Compose(
		[
		torchvision.transforms.Resize((1024,1024)),
		torchvision.transforms.ToTensor()
		]
	)

dataSet = DatasetFromFolder(data_path,transform=trans)
data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
image = next(iter(data))
#print(image[:,:1,10:200,10:200])


for i,j in enumerate(data,0):
	with torch.no_grad():
		z_ = netD(j,height=8,alpha=1)
		z_ = z_.squeeze(2).squeeze(2)
		x = netG(z_,depth=8,alpha=1)
		torchvision.utils.save_image(x, './ep9_%d_rc.jpg'%i, nrow=1)

