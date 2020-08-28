import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import PRO_GAN , Encoder , Networks as net
from pro_gan_pytorch.DataTools import DatasetFromFolder

#device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resultPath = "./result/RC_1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

resultPath1_1 = resultPath+"/imgs"
if not os.path.exists(resultPath):
    os.mkdirs(resultPath)

resultPath1_2 = resultPath+"/models"
if not os.path.exists(resultPath):
    os.mkdirs(resultPath)

#-----------------preModel-------------------
# netG = torch.nn.DataParallel(pg.Generator(depth=9))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 

# netD = torch.nn.DataParallel(pg.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netD.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

# #print(netD)
# x = torch.randn(1,3,1024,1024)
# z = netD(x,height=7,alpha=1)
# print(z.shape)

#print(gen)
# depth=0
# z = torch.randn(4,512)
# x = (gen1(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face_dp%d.jpg'%depth, nrow=4)
# del x
# x = (gen2(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face-shadow%d.jpg'%depth, nrow=4)

#---------test output------------
# netD = Encoder.encoder_v1(height=9, feature_size=512)

# #print(netD.final_block)
# x = torch.randn(1,3,1024,1024)
# z = netD(x,height=8,alpha=1)
# print(z.shape)

# netG = Networks.Generator(depth=9, latent_size=512)
# z = z.squeeze(2).squeeze(2)
# x_ = netG(z,depth=8,alpha=1)
# print(z.shape)
# print(x_.shape)


#----------------test pre-model output-----------

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
netD1 = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

netD2 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
toggle_grad(netD1,False)
toggle_grad(netD2,False)

paraDict = dict(netD1.named_parameters()) # pre_model weight dict
for i,j in netD2.named_parameters():
	if i in paraDict.keys():
		w = paraDict[i]
		j.copy_(w)

toggle_grad(netD2,True)

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
# data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
# #data_path='/Users/apple/Desktop/CelebAMask-HQ/CelebA-HQ-img'
# trans = torchvision.transforms.ToTensor()
# dataSet = DatasetFromFolder(data_path,transform=trans)
# data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=10,shuffle=True,num_workers=0,pin_memory=True)
#image = next(iter(data))
#torchvision.utils.save_image(image, './1.jpg', nrow=1)


#---------------training with true image-------------
# optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
# loss = torch.nn.MSELoss()
# loss_all=0
# for epoch in range(30):
# 	for (i, batch) in enumerate(data):
# 		image = batch.to(device)
# 		z = netD2(image,height=8,alpha=1)
# 		z = z.squeeze(2).squeeze(2)
# 		x_ = netG(z,depth=8,alpha=1)
# 		optimizer.zero_grad()
# 		loss_i = loss(x_,image)
# 		loss_i.backward()
# 		optimizer.step()
# 		print(loss_i.item())
# 		loss_all +=loss_i.item()
# 		print('loss_all'+str(loss_all))
# 		if i % 100 == 0: 
# 			torchvision.utils.save_image(image[:8], resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
# 			torchvision.utils.save_image(x_[:8], resultPath1_1+'/%d_rc.jpg'%(epoch,i), nrow=8)
# 	if epoch%10==0 or epoch == 29:
# 		torch.save(netG.state_dict(), resultPath1_2+'/G_model.pth')
# 		torch.save(netD2.state_dict(), resultPath1_2+'/G_model.pth')


#--------------training with generative image------------
optimizer = torch.optim.Adam(netD2.parameters(), lr=0.001 ,betas=(0, 0.99), eps=1e-8)
loss = torch.nn.MSELoss()
loss_all=0
for epoch in range(10):
	for i in range(100):
		z = torch.randn(10, 512).to(device)
		with torch.no_grad():
			x = netG(z,depth=8,alpha=1)
		z_ = netD2(x.detach(),height=8,alpha=1)
		z_ = z_.squeeze(2).squeeze(2)
		x_ = netG(z_,depth=8,alpha=1)
		optimizer.zero_grad()
		loss_i = loss(x_,x)
		loss_i.backward()
		optimizer.step()
		print(loss_i.item())
		loss_all +=loss_i.item()
		print('loss_all__:  '+str(loss_all))
		if i % 100 == 0: 
			img = torch.cat((x[:8],x_[:8]))
			torchvision.utils.save_image(img, resultPath1_1+'/ep%d_%d.jpg'%(epoch,i), nrow=8)
			#torchvision.utils.save_image(x_[:8], resultPath1_1+'/%d_rc.jpg'%(epoch,i), nrow=8)
	if epoch%10==0 or epoch == 29:
		torch.save(netG.state_dict(), resultPath1_2+'/G_model.pth')
		torch.save(netD2.state_dict(), resultPath1_2+'/G_model.pth')




















