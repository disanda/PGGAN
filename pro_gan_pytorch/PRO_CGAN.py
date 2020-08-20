# ========================================================================================
# ConditionalProGAN Module
# ========================================================================================

class ConditionalProGAN:
    """ Wrapper around the Generator and the Conditional Discriminator """

    def __init__(self, num_classes, depth=7, latent_size=512,
                 learning_rate=0.001, beta_1=0, beta_2=0.99,
                 eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=th.device("cpu")):
        """
        constructor for the class
        :param num_classes: number of classes required for the conditional gan
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator
                         (Used only if loss is wgan or wgan-gp)
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of ConditionalGANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam
        from torch.nn import DataParallel

        # Create the Generator and the Discriminator
        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)
        self.dis = ConditionalDiscriminator(num_classes, height=depth,feature_size=latent_size,use_eql=use_eql).to(device)

        # if code is to be run on GPU, we can use DataParallel:
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.num_classes = num_classes  # required for matching aware
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

        # setup the ema for the generator
        if self.use_ema:
            from pro_gan_pytorch.CustomLayers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_loss(self, loss):
        import pro_gan_pytorch.Losses as losses

        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = losses.CondWGAN_GP(self.dis, self.drift, use_gp=False)
                # note if you use just wgan, you will have to use weight clipping
                # in order to prevent gradient exploding

            elif loss == "wgan-gp":
                loss = losses.CondWGAN_GP(self.dis, self.drift, use_gp=True)

            elif loss == "lsgan":
                loss = losses.CondLSGAN(self.dis)

            elif loss == "lsgan-with-sigmoid":
                loss = losses.CondLSGAN_SIGMOID(self.dis)

            elif loss == "hinge":
                loss = losses.CondHingeGAN(self.dis)

            elif loss == "standard-gan":
                loss = losses.CondStandardGAN(self.dis)

            elif loss == "relativistic-hinge":
                loss = losses.CondRelativisticAverageHingeGAN(self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, losses.ConditionalGANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, labels, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param labels: (conditional classes) should be a list of integers
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss value
        """

        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples,labels, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, real_batch, labels, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: real batch of samples (real samples)
        :param labels: labels for conditional discrimination
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        # create batch of real samples
        real_samples = self.__progressive_downsampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        # TODO_complete:
        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, labels, depth, alpha)

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
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),normalize=True, scale_each=True)

    @staticmethod
    def __save_label_info_file(label_file, labels):
        """
        utility method for saving a file with labels
        :param label_file: path to the file to be written
        :param labels: label tensor
        :return: None (writes file to disk)
        """
        # write file with the labels written one per line
        with open(label_file, "w") as fp:
            for label in labels:
                fp.write(str(label.item()) + "\n")

    def one_hot_encode(self, labels):
        """
        utility method to one-hot encode the labels
        :param labels: tensor of labels (Batch)
        :return: enc_label: encoded one_hot label
        """
        if not hasattr(self, "label_oh_encoder"):
            self.label_oh_encoder = th.nn.Embedding(self.num_classes, self.num_classes)
            self.label_oh_encoder.weight.data = th.eye(self.num_classes)

        return self.label_oh_encoder(labels.view(-1))

    def train(self, dataset, epochs, batch_sizes,
              fade_in_percentage, start_depth=0, num_workers=3, feedback_factor=50,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1):
        """
        Utility method for training the ProGAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.
        :param dataset: object of the dataset used for training.
                        Note that this is not the dataloader (we create dataloader in this method
                        since the batch_sizes for resolutions can be different).
                        Get_item should return (Image, label) in that order
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution
                                   used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
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
        from pro_gan_pytorch.DataTools import get_data_loader
        print("#####")
        print(self.depth)
        print(len(batch_sizes))
        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        temp_data_loader = get_data_loader(dataset, batch_sizes[0], num_workers=3)
        _, fx_labels = next(iter(temp_data_loader))
        # reshape them properly
        fixed_labels = self.one_hot_encode(fx_labels.view(-1, 1)).to(self.device)
        fixed_input = th.randn(fixed_labels.shape[0],self.latent_size - self.num_classes).to(self.device)
        fixed_input = th.cat((fixed_labels, fixed_input), dim=-1)
        del temp_data_loader  # delete the temp data_loader since it is not required anymore

        os.makedirs(sample_dir, exist_ok=True)  # make sure the directory exists
        self.__save_label_info_file(os.path.join(sample_dir, "labels.txt"), fx_labels)

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):

            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            ticker = 1

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))

                fader_point = int((fade_in_percentage[current_depth] / 100)* epochs[current_depth] * total_batches)

                step = 0  # counter for number of iterations

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.view(-1, 1)

                    # create the input to the Generator
                    label_information = self.one_hot_encode(labels).to(self.device)
                    latent_vector = th.randn(images.shape[0],self.latent_size - self.num_classes).to(self.device)
                    gan_input = th.cat((label_information, latent_vector), dim=-1)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, labels, current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, labels, current_depth, alpha)

                    # provide a loss feedback
                    if i % int(feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                              % (elapsed, i, dis_loss, gen_loss))

                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) +"\t" + str(gen_loss) + "\n")

                        # create a grid of samples and save it
                        os.makedirs(sample_dir, exist_ok=True)
                        gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +"_" + str(epoch) + "_" +str(i) + ".png")

                        # this is done to allow for more GPU space
                        self.gen_optim.zero_grad()
                        self.dis_optim.zero_grad()
                        with th.no_grad():
                            self.create_grid(
                                samples=self.gen(fixed_input,current_depth,alpha) if not self.use_ema else self.gen_shadow(fixed_input,current_depth,alpha),
                                scale_factor=int(np.power(2, self.depth - current_depth - 1)),
                                img_file=gen_img_file)

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(save_dir,"GAN_GEN_OPTIM_" + str(current_depth)+ ".pth")
                    dis_optim_save_file = os.path.join(save_dir,"GAN_DIS_OPTIM_" + str(current_depth)+ ".pth")

                    th.save(self.gen.state_dict(), gen_save_file)
                    th.save(self.dis.state_dict(), dis_save_file)
                    th.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    th.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + ".pth")
                        th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # put the gen, shadow_gen and dis in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

        print("Training completed ...")