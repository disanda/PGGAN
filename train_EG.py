#这个版本只需要导入网络即可(不需要导入训练网络)，先已完成两个实验，第一个实验完成gt编码的比较，第二个实验完成G(z)的编码比较
#准备做 不同网络的比较，包括结构不同，weight不同的情况 (mnist中以上因素不同，区别不大)
import torch
# import numpy as np
# import os
# import torchvision
from pro_gan_pytorch import  Encoder , Networks as net1
import pro_gan_pytorch.AE as net2
# from pro_gan_pytorch.DataTools import DatasetFromFolder
# from torch.autograd import Variable

# #device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #----------------path setting---------------
resultPath = "./result/RC_Training_GD_V1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath1_1):
    os.mkdir(resultPath1_1)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath1_2):
    os.mkdir(resultPath1_2)



#----------------test pre-model output-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

netG1 = torch.nn.DataParallel(net1.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG1.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
netD1 = torch.nn.DataParallel(net1.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

netG2 = torch.nn.DataParallel(net2.Decoder_v1(depth=9,latent_size=512))
netD2 = torch.nn.DataParallel(net2.Encoder_v1(height=9, feature_size=512)) #新结构，不需要参数 

toggle_grad(netD1,False)
toggle_grad(netD2,False)
paraDict = dict(netD1.named_parameters()) # pre_model weight dict
for i,j in netD2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)
	else:
		print(i)
		print(j.requires_grad) #改动前
		j.requires_grad_(True)
		print(j.requires_grad) #改动后


toggle_grad(netG1,False)
toggle_grad(netG2,False)
paraDict = dict(netG1.named_parameters())
for i,j in netG2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)
	else:
		print(i)
		print(j.requires_grad) #改动前
		j.requires_grad_(True)
		print(j.requires_grad) #改动后

# x = torch.randn(1,3,1024,1024)
# z = netD2(x,height=8,alpha=1)
# z = z.squeeze(2).squeeze(2)
# print(z.shape)
# x_r = netG(z,depth=8,alpha=1)
# print(x_r.shape)

# z = torch.randn(5,512)
# x = (netG(z,depth=8,alpha=1)+1)/2
# torchvision.utils.save_image(x, './recons.jpg', nrow=5)

#------------------dataSet-----------
data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
#data_path='/Users/apple/Desktop/CelebAMask-HQ/CelebA-HQ-img'
trans = torchvision.transforms.ToTensor()
dataSet = DatasetFromFolder(data_path,transform=trans)
data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=10,shuffle=True,num_workers=4,pin_memory=True)

# image = next(iter(data))
# torchvision.utils.save_image(image, './1.jpg', nrow=1)


#--------------training with generative image------------: training G with D
#optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
import itertools
optimizer = optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, netG2.parameters()), filter(lambda p: p.requires_grad, netD2.parameters())),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)
loss = torch.nn.MSELoss()
loss_all=0
for epoch in range(10):
	for i in range(5001):
		z = torch.randn(10, 512).to(device)
		x = netG(z,depth=8,alpha=1)
		z_ = netD2(x.detach(),height=8,alpha=1)
		#z_ = netD2(x.detach()) #new_small_Net
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG(z_,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_i = loss(x_,x)
		loss_i.backward()
		optimizer.step()
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all)+'     loss_i:    '+str(loss_i.item()))
		if i % 100 == 0: 
			img = (torch.cat((x[:8],x_[:8]))+1)/2
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
			#torchvision.utils.save_image(x_[:8], resultPath1_1+'/%d_rc.jpg'%(epoch,i), nrow=8)
	#if epoch%10==0 or epoch == 29:
	torch.save(netG.state_dict(), resultPath1_2+'/G_model_ep%d.pth'%epoch)
	torch.save(netD2.state_dict(), resultPath1_2+'/D_model_ep%d.pth'%epoch)




















