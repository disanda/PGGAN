import torch 
import torchvision 
import sys
sys.path.append('pro_gan_pytorch')
#import PRO_GAN_Again as pg
import GAN_Encoder as pg
import torchvision
from pro_gan_pytorch.DataTools import DatasetFromFolder
import Networks as net
import Encoder

# select the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#data_path = "cifar-10/"

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

if __name__ == '__main__':
    # some parameters:
    depth = 9 # 4-->8-->16-->32-->64-->128-->256-->512-->1024 ，0开始,8结束,所以depth是9
    # hyper-parameters per depth (resolution)
    num_epochs = [10, 15, 20, 20,20,20,20,10,10]
    fade_ins = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    batch_sizes = [128, 128, 128, 128, 64, 64, 64, 4, 1]
    latent_size = 512

#----------------------配置预训练模型------------------
    netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=512))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
    netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location='cpu')) #shadow的效果要好一些 
    netD1 = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
    #netD.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location='cpu'))

    netD2 = torch.nn.DataParallel(Encoder.encoder_v1(height=9, feature_size=512))
    toggle_grad(netD1,False)
    toggle_grad(netD2,False)

    paraDict = dict(netD1.named_parameters()) # pre_model weight dict
    for i,j in netD2.named_parameters():
        if i in paraDict.keys():
            w = paraDict[i]
            j.copy_(w)

    toggle_grad(netD2,True)

    del netD1
#print(netG)
#print(netD1)


    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ProGAN(netG, netD2, depth=depth, latent_size=latent_size, device=device, use_ema=False)

    #data_path='/home/disanda/Desktop/dataSet/CelebAMask-HQ/CelebA-HQ-img'
    data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
    trans = torchvision.transforms.ToTensor()
    dataset = DatasetFromFolder(data_path,transform=trans)

    # This line trains the PRO-GAN
    pro_gan.train(
        dataSet = dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        sample_dir="./result/celeba1024-encoder/sample/",
        log_dir="./result/celeba1024-encoder/log/", 
        save_dir="./result/celeba1024-encoder/model/",
        num_workers=0,
        start_depth=8,
    )
    # ====================================================================== 

# if __name__ == '__main__':
#     # some parameters:
#     depth = 9 # 4-->8-->16-->32-->64-->128-->256-->512-->1024 ，0开始,8结束,所以depth是9
#     # hyper-parameters per depth (resolution)
#     num_epochs = [10, 10, 10, 10, 10, 10, 10, 10, 10]
#     fade_ins = [50, 50, 50, 50, 50, 50, 50, 50, 50]
#     batch_sizes = [128, 128, 128, 128, 64, 64, 64, 32, 32]
#     latent_size = 512

#     # ======================================================================
#     # This line creates the PRO-GAN
#     # ======================================================================
#     pro_gan = ProGAN(depth=depth, latent_size=latent_size, device=device)

#     #data_path='/home/disanda/Desktop/dataSet/CelebAMask-HQ/CelebA-HQ-img'
#     data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
#     #data_path='F:/dataSet2/CelebAMask-HQ/CelebA-HQ-img'
#     trans = torchvision.transforms.ToTensor()
#     dataset = DatasetFromFolder(data_path,transform=trans)

#     # This line trains the PRO-GAN
#     pro_gan.train(
#         dataSet = dataset,
#         epochs=num_epochs,
#         fade_in_percentage=fade_ins,
#         batch_sizes=batch_sizes,
#         sample_dir="./result/celeba1024_rc/",
#         log_dir="./result/celeba1024_rc/", 
#         save_dir="./result/celeba1024_rc/",
#         num_workers=0
#     )
#     # ======================================================================