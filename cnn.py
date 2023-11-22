import torch
import torch.nn as nn
from albumentations_1 import albumenataions 
from imp_funct import convert_file_name_to_numpy_array
import matplotlib.pyplot as plt 
from pandas.core.common import flatten
import copy 
import numpy as np
import pandas as pd 
import random
from torch.nn import Linear , ReLU , CrossEntropyLoss,Sequential , Conv2d , Module , Softmax,BatchNorm2d , Dropout,MaxPool2d
from torch.optim import Adam , SGD
from torch.utils.data import Dataset ,DataLoader 
import cv2
from torch.autograd import Variable
from tqdm import tqdm
import os 
input_directory="dataset-3"
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size=724 #256*256
#hidden_size=100
num_classes =3
num_epoch=10
batch_size=25
learning_rate=0.009


#create train , validation , test sets

train_data_path_idly=f'{input_directory}/dosa/dosa_file_names.csv'
train_data_path_dosa=f"{input_directory}/idly/idly_file_names.csv"
train_data_path_vada=f"{input_directory}/vada/vada_file_names.csv"
#train_data_path_others=f"{input_directory}/all vegetarian food that should not include idly,dosa and,vada/others.csv"
#for creating validation set 

# for evaluating the model
from sklearn.metrics import accuracy_score

# loading dataset
file_idly=pd.read_csv(f"{train_data_path_idly}")
file_vada=pd.read_csv(f"{train_data_path_vada}")
file_dosa=pd.read_csv(f"{train_data_path_dosa}")
#file_others=pd.read_csv(f"{train_data_path_others}")
#file_others=pd.read_csv("")
train_prcnt_image=0.6
valid_prcnt_image=0.3
test_prcnt_image=0.1
train_file_data=pd.DataFrame()
valid_file_data=pd.DataFrame()
test_file_data=pd.DataFrame()
for i in [file_idly,file_vada,file_dosa]:#,file_others
        total_images_files_selected=int(train_prcnt_image*len(i))
        train_file_data=pd.concat([train_file_data,i.iloc[0:total_images_files_selected]])
        range2=int(valid_prcnt_image*len(i))
        valid_file_data=pd.concat([valid_file_data,i.iloc[total_images_files_selected:total_images_files_selected+range2-1]])
        total_images_files_selected+=range2
        test_file_data=pd.concat([test_file_data,i.iloc[total_images_files_selected:len(i)-1]])
train_file_data.reset_index(drop=True)
valid_file_data.reset_index(drop=True)
test_file_data.reset_index(drop=True)

#sending data for data augmentation 
image_numpy_array,train_file_data=albumenataions(train_file_data)
train_x=image_numpy_array
train_y=np.array(train_file_data)#contains the value of the label
print("length of train_x" ,len(train_x))
print("length of train y",len(train_y) )

print(valid_file_data)
valid_x=convert_file_name_to_numpy_array(valid_file_data)
valid_y=np.array(valid_file_data.iloc[:,1])#contains the value of the label
print("length of valid_x" ,len(valid_x))
print("length of valid_y",len(valid_y))

test_x=convert_file_name_to_numpy_array(test_file_data)
test_y=np.array(test_file_data.iloc[:,1])#contains the value of the label
##vizualize images here 
print("length of test_x" ,len(test_x))
print("length of test_y",len(test_y))


#converting training image  and labels value to torch format
train_x=train_x.reshape(len(train_x),1,32,32)
train_x=torch.from_numpy(train_x)
class_mapping = {1: 0, 2: 1, 3: 2}
train_y=train_y.astype(int)
train_y=np.array([class_mapping[label] for label in train_y])

train_y = torch.from_numpy(train_y)
print("shape of training data")



#converting validaion image and labels value to torch format
print(valid_x.shape)
valid_x=valid_x.reshape(len(valid_x),1,32,32)

valid_x=torch.from_numpy(valid_x)


valid_y=valid_y.astype(int)
valid_y=np.array([class_mapping[label] for label in valid_y])

valid_y = torch.from_numpy(valid_y)
print(valid_y)

#converting validaion image and labels value to torch format
test_x=test_x.reshape(len(test_x),1,32,32)
test_x=torch.from_numpy(test_x)

test_y=test_y.astype(int)
test_y=np.array([class_mapping[label] for label in test_y])

test_y = torch.from_numpy(test_y)

train_dataloader=DataLoader(list(zip(train_x, train_y)),batch_size=int(len(train_y)/3),shuffle=True)
test_dataloader=DataLoader(list(zip(test_x, test_y)),batch_size=int(len(train_y)/5),shuffle=True)

class Net(nn.Module):
    def __init__(self, num_classes):

        super(Net, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(32*5*5, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out  
    


model=Net(3)
#define the optimizer 
optimizer=Adam(model.parameters(),lr=0.3,weight_decay = 0.005)
criterion=nn.CrossEntropyLoss()
total_step = len(train_dataloader)
if torch.cuda.is_available():
    model=model.cuda()
    criterion=criterion.cuda()

print(model)

num_epochs=300
device_1=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_dataloader):  
        # Move tensors to the configured device
        images = images.to(device_1)
        labels = labels.to(device_1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the {} train images: {} %'.format("s0s", 100 * correct / total))

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            print(images.shape)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the {}  images: {} %'.format("testing", 100 * correct / total))




# def train(epoch):
#     model.train()
#     tr_loss=0
#     #getting the training set 
#     x_train,y_train=Variable(train_x,requires_grad=True),Variable(train_y)
#     # getting the validation set
#     x_valid, y_valid = Variable(valid_x), Variable(valid_y)
#     #converting the data into GPU format if available

#     if torch.cuda.is_available():
#          x_train=x_train.cuda()
#          y_train=y_train.cuda()
#          x_valid=x_valid.cuda()
#          y_valid=y_valid.cuda()
#     # clearing the Gradients of the model parameters
#     optimizer.zero_grad()
#     #prediction for training and validation set 
#     for x_train,y_train in train_dataloader:
#         output_train=model(x_train)
#         print("output train",output_train)
#         #output_val=model(x_valid)

#         # computing the training and validation loss
#         loss_train=criterion(output_train, y_train)
        
#         #loss_val=criterion(output_val,y_valid)
#         train_losses.append(loss_train)
#         #val_losses.append(loss_val)
#         optimizer.zero_grad()
#         loss_train.backward()
#         optimizer.step()
#         tr_loss=loss_train.item()
#     if epoch%2==0:
#             print("Epoch :",epoch+1,"\t",'loss :',loss_train)

# # defining the number of epochs
# n_epochs = 20
# # empty list to store training losses
# train_losses = []
# # empty list to store validation losses
# val_losses = []
# # training the model
# for epoch in range(n_epochs):
#     train(epoch)

# # print(train_losses)
# # print(type(train_losses))
# #plotting the training and validation loss
# # plt.plot(train_losses, label='Training loss')
# # plt.plot(val_losses, label='Validation loss')
# # plt.legend()
# # plt.show()

# with torch.no_grad():
#     try:
#        output = model(train_x.cuda())
#     except:
#         output=model(train_x.cpu())
    
# softmax = torch.exp(output).cpu()
# prob = list(softmax.numpy())
# predictions = np.argmax(prob, axis=1)

# # accuracy on training set
# print(accuracy_score(train_y, predictions))

# with torch.no_grad():
#     try:
#        output = model(valid_x.cuda())
#     except:
#         output=model(valid_x.cpu())
    
# softmax = torch.exp(output).cpu()
# prob = list(softmax.numpy())
# predictions = np.argmax(prob, axis=1)

# # accuracy on validation set
# print(accuracy_score(valid_y, predictions))

# with torch.no_grad():
#     try:
#        output = model(test_x.cuda())
#     except:
#         output=model(test_x.cpu())
    
# softmax = torch.exp(output).cpu()
# prob = list(softmax.numpy())
# predictions = np.argmax(prob, axis=1)
# print(test_y ,"   ", predictions)

# # accuracy on training set
# print(accuracy_score(test_y, predictions))





