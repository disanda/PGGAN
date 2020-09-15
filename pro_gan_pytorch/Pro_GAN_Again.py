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
import sys
sys.path.append('pro_gan_pytorch')
from CustomLayers import _equalized_conv2d, GenGeneralConvBlock, GenInitialBlock, DisGeneralConvBlock, DisFinalBlock
from torchvision.utils import save_image
import Networks as net

#torch.backends.cudnn.benchmark = True

# function to calculate the Exponential moving averages for the Generator weights, This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """
    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)
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

#----------------------------load pre-model-------------
device= 'cuda'
netG = torch.nn.DataParallel(net.Generator(depth=9,latent_size=1024))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
#netG.load_state_dict(torch.load('./result/celeba1024/model/GAN_GEN_SHADOW_7.pth',map_location=device))
#netG.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./result/celeba1024/model/GAN_GEN_SHADOW_7.pth').items()}) #去掉一个"module"

#删除多余的<<键名>>
from collections import OrderedDict
state_dict1 = torch.load('./result/celeba1024/model/GAN_GEN_SHADOW_7.pth',map_location=device)
new_state_dict1 = OrderedDict()
for k, v in state_dict1.items():
    name = k[7:] # remove `module.`
    new_state_dict1[name] = v
netG.load_state_dict(new_state_dict1)

netD = torch.nn.DataParallel(net.Discriminator(height=9, feature_size=1024))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
#netD.load_state_dict(torch.load('./result/pre-model/GAN_DIS_3.pth',map_location=device))
#netD.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./result/pre-model/GAN_DIS_3.pth').items()})

state_dict2 = torch.load('./result/celeba1024/model/GAN_DIS_7.pth',map_location=device)
new_state_dict2 = OrderedDict()
for k, v in state_dict2.items():
    name = k[7:] # remove `module.`
    new_state_dict2[name] = v
netD.load_state_dict(new_state_dict2)

#------------------------- ProGAN Module (Unconditional)
class ProGAN:
    """ Wrapper around the Generator and the Discriminator """
    def __init__(self, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
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
        self.dis = copy.deepcopy(netD)
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
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,betas=(beta_1, beta_2), eps=eps)
        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,betas=(beta_1, beta_2), eps=eps)
        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)
        if self.use_ema:                        #复制之前模块的参数
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the, weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)
    def __setup_loss(self, loss):
        import pro_gan_pytorch.Losses as losses
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = losses.WGAN_GP(self.dis, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding
                # check the optimize_discriminator method where this has been
                # taken care of.
            elif loss == "wgan-gp":
                loss = losses.WGAN_GP(self.dis, self.drift, use_gp=True)
            elif loss == "standard-gan":
                loss = losses.StandardGAN(self.dis)
            elif loss == "lsgan":
                loss = losses.LSGAN(self.dis)
            elif loss == "lsgan-with-sigmoid":
                loss = losses.LSGAN_SIGMOID(self.dis)
            elif loss == "hinge":
                loss = losses.HingeGAN(self.dis)
            elif loss == "relativistic-hinge":
                loss = losses.RelativisticAverageHingeGAN(self.dis)
            else:
                raise ValueError("Unknown loss function requested")
        elif not isinstance(loss, losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")
        return loss
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
    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)
        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()
            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)
            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()
            loss_val += loss.item()
        return loss_val / self.n_critic
    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """
        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)
        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)
        # TODO_complete:
        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)
        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()
        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)
        # return the loss value
        return loss.item()
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
        #save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),normalize=True, scale_each=True)
        save_image(samples, img_file, nrow=8,normalize=True, scale_each=True)
    def train(self, epochs, batch_sizes,
              fade_in_percentage, num_samples=64,
              start_depth=0, num_workers=4, feedback_factor=100,
              dataSet=None, log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1):
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
                    # extract current batch of data for training
                    images = batch.to(self.device)
                    # print('img_size')
                    # print(images.shape) 这里还是1024
                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)
                    # optimize
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha)
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                    # provide a loss feedback
                    if i % 101 == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f" % (elapsed, i, dis_loss, gen_loss))
                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) + "\t" + str(gen_loss) + "\n")
                        # create a grid of samples and save it
                        os.makedirs(sample_dir, exist_ok=True)
                        gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +"_" + str(epoch) + "_" +str(i) + ".png")
                        # this is done to allow for more GPU space
                        with torch.no_grad():
                            #samples = self.gen_shadow(fixed_input,current_depth,alpha).detach()
                            #print(samples.shape)
                            self.create_grid(samples=self.gen(fixed_input,current_depth,alpha).detach() if not self.use_ema else self.gen_shadow(fixed_input,current_depth,alpha).detach(),
                                scale_factor=int(np.power(2, self.depth - current_depth - 1)),img_file=gen_img_file)
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                #if epoch % checkpoint_factor == 1 or epoch == epochs[current_depth]:
                if epoch % 1 == 0:
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
