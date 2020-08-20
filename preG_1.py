class GNet(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    def forward(self, x):
        ## Normalize the input ?
        if self.model.normalizationLayer is not None:
            x = self.model.normalizationLayer(x)
        x = x.view(-1, num_flat_features(x))#20,512
        # format layer
        x = self.model.leakyRelu(self.model.formatLayer(x))#20,8192
        x = x.view(x.size()[0], -1, 4, 4) #[20,3,512,512]
        x = self.model.normalizationLayer(x)
        # Scale 0 (no upsampling)
        for convLayer in self.model.groupScale0:
            x = self.model.leakyRelu(convLayer(x))
            if self.model.normalizationLayer is not None:
                x = self.model.normalizationLayer(x)
                # Dirty, find a better way
        if self.model.alpha > 0 and len(self.model.scaleLayers) == 1:
            y = self.model.toRGBLayers[-2](x)
            y = Upscale2d(y)
        # Upper scales
        for scale, layerGroup in enumerate(self.model.scaleLayers, 0):
            x = Upscale2d(x)
            for convLayer in layerGroup:
                x = self.model.leakyRelu(convLayer(x))
                if self.model.normalizationLayer is not None:
                    x = self.model.normalizationLayer(x)
            if self.model.alpha > 0 and scale == (len(self.model.scaleLayers) - 2):
                y = self.model.toRGBLayers[-2](x)
                y = Upscale2d(y)
        # To RGB (no alpha parameter for now)
        x = self.model.toRGBLayers[-1](x)
        # Blending with the lower resolution output when alpha > 0
        if self.model.alpha > 0:
            x = self.model.alpha * y + (1.0-self.model.alpha) * x
        if self.model.generationActivation is not None:
            x = self.model.generationActivation(x)
        return x