import torch.nn as nn


# Number of channels in the training images
nc = 3
# Size of latent vector in Discriminator
nz = 512
# Size of generator channels
ngc = 64

class StegEncoder(nn.Module):
    def __init__(self):
        super(StegEncoder, self).__init__()

        self.main = nn.Sequential(
            #input is (2*nc) x 256 x 256
            nn.Conv2d(2*nc, ngc, 7, 1, 3, bias = True),
            nn.BatchNorm2d(ngc),
            nn.ReLU(True),
            #state size: ngc x 256 x 256
            nn.Conv2d(ngc, 2 * ngc, 3, 2, 1, bias = True),
            nn.BatchNorm2d(2 * ngc),
            nn.ReLU(True),
            #state size: 2 * ngc x 128 x 128
            nn.Conv2d(2 * ngc, 4 * ngc, 3, 2, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            #state size: 4 * ngc x 64 x 64
        )
        
        self.res_block1 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block2 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block3 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block4 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block5 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block6 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block7 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block8 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.res_block9 = nn.Sequential(
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            nn.Conv2d(4 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
        )

        self.final = nn.Sequential(
            #input is 4 * ngc x 64 x 64
            nn.ConvTranspose2d(4 * ngc, 2 * ngc, 3, 2, 1, 1, bias = True),
            nn.BatchNorm2d(2 * ngc),
            nn.ReLU(True),
            #state size: 2 * ngc x 128 x 128
            nn.ConvTranspose2d(2 * ngc, ngc, 3, 2, 1, 1, bias = True),
            nn.BatchNorm2d(ngc),
            nn.ReLU(True),
            #state size: ngc x 256 x 256
            nn.Conv2d(ngc, nc, 7, 1, 3, bias = True),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.main(input)
        x = self.res_block1(x) + x
        x = self.res_block2(x) + x
        x = self.res_block3(x) + x
        x = self.res_block4(x) + x
        x = self.res_block5(x) + x
        x = self.res_block6(x) + x
        x = self.res_block7(x) + x
        x = self.res_block8(x) + x
        x = self.res_block9(x) + x
        x = self.final(x)
        return x

class StegDecoder(nn.Module):
    def __init__(self):
        super(StegDecoder, self).__init__()

        self.main = nn.Sequential(
            #input is nc x 256 x 256
            nn.Conv2d(nc, ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(ngc),
            nn.ReLU(True),
            #state size: ngc x 256 x 256
            nn.Conv2d(ngc, 2 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(2 * ngc),
            nn.ReLU(True),
            #state size: 2 * ngc x 256 x 256
            nn.Conv2d(2 * ngc, 4 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(4 * ngc),
            nn.ReLU(True),
            #state size: 4 * ngc x 256 x 256
            nn.Conv2d(4 * ngc, 2 * ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(2 * ngc),
            nn.ReLU(True),
            #state size: 2 * ngc x 256 x 256
            nn.Conv2d(2 * ngc, ngc, 3, 1, 1, bias = True),
            nn.BatchNorm2d(ngc),
            nn.ReLU(True),
            #state size: ngc x 256 x 256
            nn.Conv2d(ngc, nc, 3, 1, 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.compress = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ngc, 4, 2, 1, bias = True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngc) x 128 x 128
            nn.Conv2d(ngc, ngc * 2, 4, 2, 1, bias = True),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngc*2) x 64 x 64
            nn.Conv2d(ngc * 2, ngc * 2, 4, 2, 1, bias = True),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngc*2) x 32 x 32
            nn.Conv2d(ngc * 2, ngc * 2, 4, 2, 1, bias = True),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. (ngc*2) x 16 x 16
            nn.AvgPool2d(4),
            #state size. (ngc*2) x 4 x 4
            nn.Flatten(),
            nn.Linear(ngc*2*4*4, nz),
        )

        self.classify = nn.Sequential(
            #input is nz
            nn.Linear(int(nz), int(nz / 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(nz / 8), int(1)),
            nn.Sigmoid()
        )

    def forward(self, image):
        image = self.compress(image)
        image = self.classify(image)
        return image