import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch
import torchvision
from torch.nn import ModuleList, Conv2d, AvgPool2d, DataParallel
from torch.nn.functional import interpolate
from torch.optim import Adam
from torchvision.utils import save_image
import sys
sys.path.append('pro_gan_pytorch')
from CustomLayers import _equalized_conv2d, GenGeneralConvBlock, GenInitialBlock, DisGeneralConvBlock, DisFinalBlock
import Networks as net
import Encoder

resultPath = "./result_encoder_1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

# utility function for toggling the gradient requirements of the models
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

# function to calculate the Exponential moving averages for the Generator weights, This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """
    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)

device= 'cuda'

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

#ProGAN Module (Unconditional)
class ProGAN:
    """ Wrapper around the Generator and the Discriminator """
    def __init__(self,netG,netD, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=torch.device("cpu")):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator per generator update
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """
        # Create the Generator and the Discriminator
        self.gen = copy.deepcopy(netG)
        self.dis = copy.deepcopy(netD2)
        del netG,netD2
        # if code is to be run on GPU, we can use DataParallel:
        if device == torch.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)
        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift
        # define the optimizers for the discriminator and generator
        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,betas=(beta_1, beta_2), eps=eps)
        # define the loss function used for training the GAN
        self.loss = torch.nn.MSELoss()
        if self.use_ema:                        #复制之前模块的参数
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the, weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)
    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the progressive growing of the layers. 将原图下采样为对于阶段的分辨率
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """
        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth-1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)
        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)
        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples
        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)
        return real_samples
    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)
        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),normalize=True, scale_each=True)
    def train(self, epochs, batch_sizes,
              fade_in_percentage, num_samples=64,
              start_depth=0, num_workers=4, feedback_factor=100,
              dataSet=None, log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=10):
        """
        Utility method for training the ProGAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.
        :param dataset: object of the dataset used for training.
                        Note that this is not the dataloader (we create dataloader in this method
                        since the batch_sizes for resolutions can be different)
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution
                                   used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param num_workers: number of workers for reading the data. def=3
        :param feedback_factor: number of logs per epoch. def=100
        :param log_dir: directory for saving the loss logs. def="./models/"
        :param sample_dir: directory for saving the generated samples. def="./samples/"
        :param checkpoint_factor: save model after these many epochs.
                                  Note that only one model is stored per resolution.
                                  during one resolution, the checkpoint will be updated (Rewritten)
                                  according to this factor.
        :param save_dir: directory for saving the models (.pth files)
        :return: None (Writes multiple files to disk)
        """
        #print('#######3')
        #print(self.depth)
        #print(len(batch_sizes))
        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"
        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()
        # create a global time counter
        global_time = time.time()
        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)
#-----------------training-------------------
        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):
            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1
            data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=batch_sizes[current_depth],shuffle=True,num_workers=num_workers,pin_memory=True)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))
                #fader_point = int((fade_in_percentage[current_depth] / 100)* epochs[current_depth] * total_batches)
                fader_point = fade_in_percentage[current_depth] / 100
                step = 0  # counter for number of iterations
                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers, alpha = ticker / fader_point if ticker <= fader_point else 1
                    alpha = fader_point if fader_point <1 else 1
                    images = batch.to(self.device)
                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)
                    # optimize
                    z = self.dis(images,height=epoch,alpha=1)
                    z = z.squeeze(2).squeeze(2)
                    x_ = self.gen(z,depth=epoch,alpha=1)
                    self.dis_optim.zero_grad()
                    loss = self.loss(x_,images)
                    loss.backward()
                    self.dis_optim.step()
                    dis_loss += loss.item()
# provide a loss feedback
                    if i % int(feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d   d_loss: %f" % (elapsed, i, dis_loss))
                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) + "\t"  + "\n")
                        # create a grid of samples and save it
                        os.makedirs(sample_dir, exist_ok=True)
                        gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +"_" + str(epoch) + "_" +str(i) + ".png")
                        # this is done to allow for more GPU space
                        with torch.no_grad():
                            self.create_grid(samples=self.gen(fixed_input,current_depth,alpha).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input,current_depth,alpha).detach(),scale_factor=int(np.power(2, self.depth - current_depth - 1)),img_file=gen_img_file)
                            torchvision.utils.save_image(real_samples, resultPath+'/recons-%d-%d.jpg'%(epoch,i), nrow=8)
                            torchvision.utils.save_image(fake_samples, resultPath+'/face-%d-%d.jpg'%(epoch,i), nrow=8)
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1
                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                if epoch % checkpoint_factor == 10 or epoch == epochs[current_depth]:
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(save_dir,"GAN_GEN_OPTIM_" + str(current_depth)+ ".pth")
                    dis_optim_save_file = os.path.join(save_dir,"GAN_DIS_OPTIM_" + str(current_depth)+ ".pth")
                    torch.save(self.gen.state_dict(), gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)
                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +str(current_depth) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
        # put the gen, shadow_gen and dis in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()
        print("Training completed ...")
