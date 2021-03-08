from model.unet3d import GANLoss, Generator, Discriminator
import torch.optim as optim
import torch
class train_network:
    def __init__(self, in_channels_G,out_channels_G, device,lr_rate,amsgrad,weight_decay,lambda_L1=100):
        self.device=device
        self.lr_rate=lr_rate
        self.amsgrad=amsgrad
        self.weight_decay=weight_decay
        self.netG = Generator(in_channels_G,out_channels_G).to(self.device)
        self.netD = Discriminator(2*out_channels_G).to(self.device)
        self.optimizer_G = optim.AdamW(self.netG.parameters(), lr=self.lr_rate, eps=1e-04, amsgrad=self.amsgrad, weight_decay=self.weight_decay)
        self.optimizer_D = optim.AdamW(self.netD.parameters(), lr=self.lr_rate, eps=1e-05, amsgrad=self.amsgrad, weight_decay=self.weight_decay)
        self.criterionGAN = GANLoss('wgangp').to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.lambda_L1=lambda_L1
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['image'].to(self.device)
        self.real_B = input['mask'].to(self.device)
#         self.image_paths = input['A_paths']
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,mode='train'):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.score = pred_fake
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
      
        
    def calculate_loss(self, real, fake, image):
        fake_AB = torch.cat((image, fake), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake, real) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        return self.loss_G.item(), self.score