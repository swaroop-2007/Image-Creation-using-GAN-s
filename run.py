#Importing the libraries
from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

#Setting some hyperparameters
#Hyperparameters :  Variables that determine the newtork structure i.e No. of hidden units
#                   And variables determining the training of network i.e Learning Rate
batch_Size = 64
image_Size = 64

#Creating the transformation
transform = transforms.Compose([transforms.Scale(image_Size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_Size, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:    #Will look for word Conv and initialize its wts as 0.0 & 0.02
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  #Will look for word BatchNorm and init its wts as 1.0 & 0.02
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Defining the Generator
class Generator(nn.Module):    #Inheriting the neural network module in generator class
    def __init__(self):        #Function for generating architecture of generator
        super(Generator, self).__init__()   #Activating inheritance of nn.Module
        #Creating meta module and defining architecture of neural network
        self.main = nn.Sequential( 
        nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),   #Starting inverse CNN
                                #Size of i/p, feature maps, size of kernel, stride, padding, no-bias
        nn.BatchNorm2d(512),  #Applying batch normalisation for feature maps
        nn.ReLU(True),         #Activating Rectifier LinearUnit
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   #Adding another inverse CNN
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  
                #Making a generator with 3 channels of fake images hence we want an output of 3 
        nn.Tanh()   #Breaking the linearity. We will also have output values between -1 and 1.
        )    

    def forward(self, input):       #Function for forward propogation. 
                #input is the random vector that will i/p the noise by which generator will generate an image

        output = self.main(input)
        return output

#Creating the generator
neuralG = Generator()     #Object of class Generator
neuralG.apply(weights_init)  #Applying the wts to our generator


#Defining the discriminator
class Discriminator(nn.Module):   #Inheriting the neural network module in discriminator class
    def __init__(self):
        super(Discriminator, self).__init__()   #Activating inheritance of nn.Module
        
        #Creating meta module and defining architecture of neural network

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),   #Starting the simple convolution. i/p channels are 3 i.e the o/p of generator class. 
                            #o/p channels are 64, kernel_size is 4*4, stride is 2, padding is 1, no bias
            nn.LeakyReLU(0.2, inplace=True),   #Applying Rectification. negative slope is 0.2, inplace is True
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),   #Increasing the o/p channels
            nn.BatchNorm2d(128),   #Applying batch normalisation for o/p feature maps.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),    #Last convolution. Last o/p will be single vector i.e 1
            nn.Sigmoid()   #Break the linearity. This activation function also outputs the values in the range of 0 & 1.
        )
    
    def forward(self, input):       #Function for forward propogation.
        output = self.main(input)
        return output.view(-1)     #Flatten the result of convolution from 2d into single dimension

#Creating the discriminator
neuralD = Discriminator()   #Object of Discriminator
neuralD.apply(weights_init)

#Training our neural network

criterion = nn.BCELoss()    #Measure the error between the predicted value( betn 0 and 1) and ground truth value( 0 and 1)
optimizerDiscriminator = optim.Adam(neuralD.parameters(), lr=0.0002, betas=(0.5,0.999)) #Creating the optimizer of discriminator. 
            #lr is learning rate, betas coefficients used for computing running averages of gradient & its square
optimizerGenerator = optim.Adam(neuralG.parameters(), lr=0.0002, betas=(0.5,0.999))

if __name__=='__main__':
    for epoch in range(25):    #Loop over different epochs. Going through all the imgs of dataset 25 times to create great images
        for i, data in enumerate(dataloader, 0):          #Going through the images in dataset. Breaking the images in mini batches. data is different mini batch
            #STEP1:Updating the wts of nn of discriminator
            neuralD.zero_grad()     #Init the gradients w.r.t the wts
            
            # Training the discriminator with real images from the dataset
            #This will train the discriminator about what is real
            real, _ = data   #_ is ignore the second data. real is a tensor
            input = Variable(real)  #Converting into torch variable
            target = Variable(torch.ones(input.size()[0]))       #Setting the target. torch.ones will give the target as 1 to all the images of dataset as 1 means the image is accepted
            ouput = neuralD(input)      #It will forward propogate the nn and the ouput will be between 0 & 1
            errorDiscriminator_real = criterion(ouput, target)  #Calculating the error between the output and the target

            #Traing the Discriminator with fake image generated by the Generator
            #This will train the discriminator about what is fake
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))   #Creating the random vector of 100 feature maps of size 1*1
            fake = neuralG(noise)      #Ground truth of fake images
            target = Variable(torch.zeros(input.size()[0]))     #Target should be the rejection of images. Hence it is a tensor full of 0
            ouput = neuralD(fake.detach())          #It will forward propogate the nn and the ouput will be between 0 & 1
            errorDiscriminator_fake = criterion(ouput, target)

            #Backpropogating the total error
            errorDiscriminator = errorDiscriminator_real + errorDiscriminator_fake   #Calculating total error
            errorDiscriminator.backward()       #Backpropogating
            optimizerDiscriminator.step()   #Update the wts according to the total error of the particular wt

            #STEP2:Updating the wts of nn of generator
            neuralG.zero_grad()     #Init the gradients w.r.t the wts
            target = Variable(torch.ones(input.size()[0]))       #Setting the target. torch.ones will give the target as 1 to all the images of dataset as 1 means the image is accepted
            ouput = neuralD(fake)
            errorGenerator = criterion(ouput, target)
            errorGenerator.backward()       #Backpropogating
            optimizerGenerator.step()       #Update the wts according to the total error of the particular wt

            #STEP3: Printing the losses and saving images in our folder
            print('[%d/%d][%d/%d] Loss_D: %.4f LossG: %.4f' % (epoch, 25, i, len(dataloader), errorDiscriminator.item(), errorGenerator.item()))
            if i % 100 == 0:    #Every 100 steps
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)   #Saving the real images
                fake = neuralG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)   #Saving the fake images
                
