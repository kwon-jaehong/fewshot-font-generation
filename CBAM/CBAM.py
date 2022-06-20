import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import torchvision
import torch.optim as optim  # Pytoch Common optimization methods are encapsulated in torch.optim Inside 
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader

model_urls = {
    
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}
#  Define convolution kernel 3*3,padding=1,stride=1 Convolution of , The characteristic of this convolution is that it does not change the size, But it may change channel
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
#  Definition CBAM Medium channel attention modular 
class ChannelAttention(nn.Module):
    #  The number of incoming input channels and compression multiple are required reduction, The default is 16
    def __init__(self,in_channels,reduction = 16):
        super(ChannelAttention,self).__init__()
        # Define the required global average pooling and global maximum pooling , Number of incoming and output channels output_size = 1
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.max = nn.AdaptiveMaxPool2d(output_size=1)
        ##  Define shared perceptron MLP
        self.fc1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels//reduction,kernel_size=1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=in_channels//reduction,out_channels=in_channels,kernel_size=1,bias=False)

        ##  Define the linear activation function 
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max(x))))

        out = self.sigmod(avg_out + max_out)

        return out

##  Definition spatial attention modular 
class SpatialAttention(nn.Module):
    #  The spatial attention module has a convolution layer , Need to kernel_size
    def __init__(self,kernel_size = 7):
        super(SpatialAttention,self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channels = 2,out_channels = 1,kernel_size=kernel_size,padding = padding,bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        #  take avg_out  and  max_out stay channel Stitching on dimensions 
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)

        return self.sigmod(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channels,out_channels,stride = 1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.ca = ChannelAttention(out_channels)

        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,block,layers,num_classes=1000):
        self.in_channels = 64
        super(ResNet,self).__init__()
        # MNIST It's a grayscale image , There's only one way , therefore in_channels = 1
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride = 2, padding = 1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        # self.Avgpool = nn.AvgPool2d(2,stride=1)
        self.fc = nn.Linear(512 * block.expansion,num_classes)
        #  Weight initialization 
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,planes * block.expansion,
                kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                
            )
        layers = []
        layers.append(block(self.in_channels,planes,stride,downsample))
        self.in_channels = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.in_channels,planes))
        
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.Avgpool(x) # MNIST Data sets 28*28, Is too small , Unsuitable for doing pooling
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return F.log_softmax(x,dim = 1)

def resnet18_cbam(pretrained = False,**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],**kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model

""" ## pretrained All the pre training parameters are input 3 Convolution of channels , Training MNIST There is no way to use pre training weights  my_model = resnet18_cbam(pretrained=False,num_classes = 10) x = torch.Tensor(10,1,28,28) print(my_model(x).size()) """
#  Network training 
def train(epoch):
    network.train()  #  use model.train() Will take all of module Set to training mode .
    #  If it is a test or verification phase , You need to keep the model in the validation phase , That is to call model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):  #  Combine a traversable data object into an index sequence , List both data and data index 
        #batch_idx  Of each training set batch id; data  The data set of each training set batch  Size = [64,1,28,28]; target  A collection of real numbers represented by each training set batch Size = [1000]
        optimizer.zero_grad()  #  Gradient cleaning , because PyTorch By default, gradients are accumulated 
        #  By default, gradients are cumulative , You need to initialize or reset the gradient manually , call optimizer.zero_grad() that will do 
        output = network(data)  # data Is the image input value , adopt network obtain out Output value 
        loss = F.nll_loss(output, target)  #  Output through the network out And the target target Compare and calculate the loss value loss
        loss.backward()  #  Error back propagation   Loss function (loss) call backward() that will do 
        optimizer.step()  #  Based on the current gradient （ Stored in the of the parameter .grad Properties of the ） Update parameters 

        if batch_idx % log_interval == 0:
            print("Train Epoch:{}[{}/{}({:.0f}]\tLoss:{:.6f})".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))  # print

            train_losses.append(loss.item())  #  This time epoch Of loss Save to train_losses list 
            train_counter.append(  #  This time epoch Of counter Save to train_counter list 
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        torch.save(network.state_dict(), './model.pth')  #  Use of neural network module .state_dict() Save and load their internal state 
        torch.save(optimizer.state_dict(), './optimizer.pth')  #  Optimizer usage .state_dict() Save and load their internal state 

#  Evaluate the performance of the output network   Output loss and accuracy
def test():
    network.eval()  #  Validation phase , You need to keep the model in the validation phase , That is to call model.eval()
    test_loss = 0   # loss Initial value 
    correct = 0     # correct Initial value 
    with torch.no_grad():  # torch.no_grad： One tensor（ Name it x） Of requires_grad = True, from x Get new tensor（ Name it w- Scalar ）requires_grad Also for the False, And grad_fn Also for the None, That's not right w Derivation 
        for data, target in test_loader:  #  Test set download generator 
            # data Is the image data of the test set ,target Is the value of the real number represented by the test set picture 
            output = network(data)  #  Computing network output  out What kind of format type is it ？
            test_loss += F.nll_loss(output, target, size_average=False).item()  #  Calculate... On the test set loss
            #  Calculate the accuracy of the classification 
            pred = output.data.max(1, keepdim=True)[1]   # pred Is the output of the predicted value of the maximum possibility for each picture 
            correct += pred.eq(target.data.view_as(pred)).sum()  #  Calculate classification accuracy 
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) # print


batch_size_train = 64
batch_size_test = 1000
#  utilize Pytorch Built in functions mnist Download data 
#  download MNIST Training set , Save in a fixed format 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))  # Normalize((0.1307,),(0.3081,)) Normalize the tensor   There's only one way   So there's only one dimension 
                               ])),
    batch_size=batch_size_train, shuffle=True)

#  download MNIST Test set , Save in a fixed format 
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))  # Normalize((0.1307,),(0.3081,)) Normalize the tensor   There's only one way   So there's only one dimension 
                                   #  Respectively represent the mean and variance normalized to the tensor 
                               ])),
    batch_size=batch_size_test, shuffle=True)
# train_loader test_loader Generator 
# torchvision.transforms.Compose Some conversion functions can be combined 
#  because MNIST The image is gray with only one channel , If there are multiple channels , Multiple numbers are required , Like three channels , Should be Normalize([m1,m2,m3], [n1,n2,n3])
# download Parameter controls whether to download , If ./data There are already MNIST, Can choose False
#  use DataLoader Get the generator , This saves memory 
n_epochs = 3  # epoch Number 
# one epoch = one forward pass and one backward pass of all the training examples
learning_rate = 0.01  #  Learning rate 

momentum = 0.5  #  momentum 
random_seed = 1  #  Random seed number 
torch.manual_seed(random_seed)  #  by CPU Set seeds to generate random numbers , So that the result is certain 




#  Implement a network 
network = resnet18_cbam(pretrained=False,num_classes = 10)
# SGD Is the most basic optimization method 
# SGD The data will be split and then put into... In batches  NN  Middle computation .
#  Use batch data every time ,  Although it does not reflect the overall data ,  But it has accelerated to a great extent  NN  Training process of ,  And you won't lose too much accuracy .
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# momentum It's a parameter   The change of the current weight will be affected by the change of the last weight ,
Step_LR = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 1,gamma=0.4)
#  Just like when the ball rolls , Because of inertia , The current state will be affected by the previous state , This can speed up .

log_interval = 10  # 10 One training batch corresponds to one training epoch
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

test()  #  Test the network performance of initializing network parameters 
# Test set: Avg. loss: 2.3004, Accuracy: 751/10000 (8%)

#  Cyclic training network 
for epoch in range(1, n_epochs + 1):
    train(epoch)  #  Train one epoch
    test()  #  Train one epoch after , Test network performance in real time , Output loss and accuracy
    Step_LR.step()  #  Start learning rate decay 

#  The output after the end of the last cycle training 
# Test set: Avg. loss: 0.0927, Accuracy: 9713/10000 (97%)

#  Draw a training curve 
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')  # x The axis is train_counter list  y The axis is train_losses list   draw a curve 
plt.scatter(test_counter, test_losses, color='red')  #  Draw a scatter plot 
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  #  Add a legend to the picture 
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()  #  Show the graph 