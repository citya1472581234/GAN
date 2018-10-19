from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
dataset = 'facades'
batchSize = 1
testBatchSize =1
nEpochs = 200
input_nc=3
output_nc=3
ngf=64
ndf=64
lr=0.0002
beta1=0.5
cuda = True 
threads=0
seed = 123
lamb=10
# weight on L1 term in objective


if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
# increase efficient 


torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + dataset)
test_set = get_test_set(root_path + dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)

print('===> Building model')
netG = define_G(input_nc, output_nc, ngf, 'batch', False, [0])
netD = define_D(input_nc + output_nc, ndf, 'batch', False, [0])

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(batchSize, input_nc, 256, 256)
real_b = torch.FloatTensor(batchSize, output_nc, 256, 256)

if cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG(real_a)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizerG.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        if cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = netG(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", dataset)):
        os.mkdir(os.path.join("checkpoint", dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + dataset))

for epoch in range(1, nEpochs + 1):
    train(epoch)
    test()
    if epoch % 50 == 0:
        checkpoint(epoch)
