import torch as th
import torchvision as tv
import sys
sys.path.append('pro_gan_pytorch')
#import PRO_GAN_Again as pg
import GAN_Endoer as pg
import torchvision
from pro_gan_pytorch.DataTools import DatasetFromFolder

# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")
#data_path = "cifar-10/"

if __name__ == '__main__':
    # some parameters:
    depth = 9 # 4-->8-->16-->32-->64-->128-->256-->512-->1024 ，0开始,8结束,所以depth是9
    # hyper-parameters per depth (resolution)
    num_epochs = [10, 15, 20, 20,20,20,20,10]
    fade_ins = [100, 90, 80, 70, 60, 50, 40, 30, 20]
    batch_sizes = [128, 128, 128, 128, 64, 64, 64, 32, 16]
    latent_size = 512
    
    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ProGAN(depth=depth, latent_size=latent_size, device=device)

    #data_path='/home/disanda/Desktop/dataSet/CelebAMask-HQ/CelebA-HQ-img'
    data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
    trans = torchvision.transforms.ToTensor()
    dataset = DatasetFromFolder(data_path,transform=trans,use_ema=False)

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