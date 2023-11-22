
import torch
import torch.nn as nn
from torchinfo import summary
from albumentations_1 import albumenataions 
import albumentations as A
import cv2

from imp_funct import convert_file_name_to_numpy_array
import matplotlib.pyplot as plt 
from pandas.core.common import flatten
import numpy as np
import pandas as pd 
from  torchvision import models,datasets,models,transforms
from torch.nn import Linear , ReLU , CrossEntropyLoss,Sequential , Conv2d , Module , Softmax,BatchNorm2d , Dropout,MaxPool2d
from torch.optim import Adam , SGD
from torch.utils.data import DataLoader
import cv2
from torch.autograd import Variable
from tqdm import tqdm
import os 
device_1=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
class_names={"dosa":file_dosa.iloc[0,1]-1,"vada":file_vada.iloc[0,1]-1,"idly":file_idly.iloc[0,1]-1}
print("this is class name", class_names)
train_prcnt_image=0.7
valid_prcnt_image=0.2
test_prcnt_image=0.1
train_file_data=pd.DataFrame()
valid_file_data=pd.DataFrame()
test_file_data=pd.DataFrame()
for i in [file_idly,file_vada,file_dosa]:#,file_others
        total_images_files_selected=int(train_prcnt_image*len(i))
        train_file_data=pd.concat([train_file_data,i.iloc[0:total_images_files_selected-1]])

        range2=int(valid_prcnt_image*len(i))
        valid_file_data=pd.concat([valid_file_data,i.iloc[total_images_files_selected:total_images_files_selected+range2-1]])

        total_images_files_selected+=range2
        test_file_data=pd.concat([test_file_data,i.iloc[total_images_files_selected:len(i)-1]])

train_file_data.reset_index(drop=True)
valid_file_data.reset_index(drop=True)
test_file_data.reset_index(drop=True)




#sending data for data augmentation 
image_numpy_array,train_file_data=albumenataions(train_file_data,task="pretrain_last_layer")
train_x=image_numpy_array
train_y=np.array(train_file_data)#contains the value of the label
print("length of train_x" ,len(train_x))
print("length of train y",len(train_y) )

valid_x=convert_file_name_to_numpy_array(valid_file_data,"pretrain_last_layer")
valid_y=np.array(valid_file_data.iloc[:,1])#contains the value of the label
print("length of valid_x" ,len(valid_x))
print("length of valid_y",len(valid_y))

test_x=convert_file_name_to_numpy_array(test_file_data,"pretrain_last_layer")
test_y=np.array(test_file_data.iloc[:,1])#contains the value of the label
##vizualize images here 
print("length of test_x" ,len(test_x))
print("length of test_y",len(test_y))


#converting training image  and labels value to torch format
train_x=train_x.reshape(len(train_x),3,230,230)
train_x=torch.from_numpy(train_x)
class_mapping = {1: 0, 2: 1, 3: 2}
train_y=train_y.astype(int)
train_y=np.array([class_mapping[label] for label in train_y])

train_y = torch.from_numpy(train_y)
print("shape of training data")



#converting validaion image and labels value to torch format
print(valid_x.shape)
valid_x=valid_x.reshape(len(valid_x),3,230,230)

valid_x=torch.from_numpy(valid_x)


valid_y=valid_y.astype(int)
valid_y=np.array([class_mapping[label] for label in valid_y])
valid_y = torch.from_numpy(valid_y)

#converting validaion image and labels value to torch format
test_x=test_x.reshape(len(test_x),3,230,230)
test_x=torch.from_numpy(test_x)

test_y=test_y.astype(int)
test_y=np.array([class_mapping[label] for label in test_y])

test_y = torch.from_numpy(test_y)

train_dataloader=DataLoader(list(zip(train_x, train_y)),batch_size=int(len(train_y)/6),shuffle=True)
test_dataloader=DataLoader(list(zip(test_x, test_y)),batch_size=int(len(test_y)/5),shuffle=True)
valid_dataloader=DataLoader(list(zip(valid_x,valid_y)),batch_size=int(len(valid_y)/3),shuffle=True)

train_data_length=len(train_x)
valid_data_length=len(valid_x)

#load pretrained resnet50 
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
i=0
for params in resnet50.parameters():
        params.requires_grad=False



unfreeze_layers = ["layer4", "fc"]#these are last two layers of the resnet50 model 
for name, param in resnet50.named_parameters():
    if any(layer_name in name for layer_name in unfreeze_layers):
        param.requires_grad = True

for name, param in resnet50.named_parameters():
    print(f"{name}: {param.requires_grad}")

in_features=resnet50.fc.in_features
out_features=3
resnet50.fc=nn.Linear(in_features,out_features)

# fc_inputs=resnet50.fc.in_features
# resnet50.fc = nn.Sequential(
# #     nn.Linear(fc_inputs, 256),
# #     nn.ReLU(),
# #     nn.Dropout(0.4),
# #     nn.Linear(256, 3), # Since 10 possible outputs
# #     nn.LogSoftmax(dim=1) # For using NLLLoss()
#      nn.Dropout(0.4),
#      nn.Linear(fc_inputs, 3),
#      nn.ReLU(),
#      nn.LogSoftmax(dim=1) )

    


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_1=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = resnet50.to(device)
#define optimizer and loss
criterion=nn.CrossEntropyLoss()
loss_func= nn.NLLLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, resnet50.parameters()), lr=0.009)



def train_and_validate( model,loss_criterion,optimizer,epochs=30):
        best_loss = 100000.0
        for epoch in range(epochs):
                print("epoch: {}/{}".format(epoch+1,epochs))
                model.train()

                #loss and accuracy within epoch .
                train_loss=0
                train_acc=0
                valid_loss=0
                valid_acc=0
                for i ,(inputs,labels) in enumerate(train_dataloader):
                        inputs=inputs.to(device)
                        labels=labels.to(device)
                        optimizer.zero_grad()
                        outputs=model(inputs)
                        loss=loss_criterion(outputs,labels)
                        loss.backward()

                        optimizer.step()
                        train_loss+=loss.item()* inputs.size(0)

                        ret, predictions = torch.max(outputs.data, 1)
                        correct_counts = predictions.eq(labels.data.view_as(predictions))
                        acc=torch.mean(correct_counts.type(torch.FloatTensor))
                        
                         # Convert correct_counts to float and then compute the mean
                        acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
                        #    Compute total accuracy in the whole batch and add to train_acc
                        train_acc += acc.item() * inputs.size(0)
                  # Validation - No gradient tracking needed
                with torch.no_grad():

            # Set to evaluation mode
                        model.eval()

                # Validation loop
                        count=0
                        for j, (inputs, labels) in enumerate(valid_dataloader):
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                if count<1:
                                

                                # Forward pass - compute outputs on input data using the model
                                        outputs = model(inputs)
                                        print("this is the length of the valid_dataloader: ", count)
                                        print("this is the output",outputs)
                                        predicted_classes = torch.argmax(outputs, dim=1)
                                        print(predicted_classes)
                                        print("this is inputs value",inputs.shape)

                                        # Compute loss
                                        loss = loss_criterion(outputs, labels)
                                        print("this is the loss",loss)

                                        # Compute the total loss for the batch and add it to valid_loss
                                        valid_loss += loss.item() * inputs.size(0)

                                        # Calculate validation accuracy
                                        ret, predictions = torch.max(outputs.data, 1)
                                        print("this is the predictions",predictions)
                                        correct_counts = predictions.eq(labels.data.view_as(predictions))
                                        print("\n\n\n",correct_counts)

                                        # Convert correct_counts to float and then compute the mean
                                        acc = torch.mean(correct_counts.type(torch.FloatTensor))

                                        # Compute total accuracy in the whole batch and add to valid_acc
                                        valid_acc += acc.item() * inputs.size(0)  
                                count+=1      

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
                if valid_loss < best_loss:
                        best_loss = valid_loss
                        best_epoch = epoch

                # Find average training loss and training accuracy
                avg_train_loss = train_loss/train_data_length
                avg_train_acc = train_acc/train_data_length

                # Find average training loss and training accuracy
                avg_valid_loss = valid_loss/valid_data_length 
                avg_valid_acc = valid_acc/valid_data_length
                history=[]
                history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100))
                torch.save(model, "preTLAstfew"+'_model_'+str(epoch)+'.pt')
        #summary(model,input_size= (32,3,32, 32),col_names=["input_size", "output_size", "num_params","trainable"],col_width=20,
        #row_settings=["var_names"])
        # # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
        
        return model , history , best_epoch

#Train the model for 25 epochs
num_epochs =30
trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

torch.save(history,  "dataset-3"+'_history.pt')


train_transforms=A.Compose(
    [
        A.Resize(width=230,height=230),
        A.RandomCrop(width=230,height=230),
        A.Rotate(limit=50,p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize([0.485, 0.456, 0.406] ,[0.229, 0.224, 0.225])
        #A.RGBShift(r_shift_limit=5,g_shift_limit=5,b_shift_limit=5,p=0.5),
        # A.OneOf([A.Blur(blur_limit=3,p=0.5),
        #          A.ColorJitter(p=0.5),
        #          ] , p=1.0)

    ]
)    
def deploy_and_test(image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display=image
       
        image=image.astype("float32")
        image=image/255

        augmenatations= train_transforms(image=image)
        image=augmenatations["image"]
        with torch.no_grad():
                trained_model.eval()
                image=torch.from_numpy(image)
                image=image.reshape(1,3,230,230)
                print(type(image))

                #print(torch.unsqueeze(image,0).shape)
                image = image.to(device_1)
                outputs = trained_model(image)
                target_image_pred_probs = torch.softmax(outputs, dim=1)
                 #9. Convert prediction probabilities -> prediction labels
                predicted_classes = torch.argmax(target_image_pred_probs, dim=1)                
                print(predicted_classes)

                if predicted_classes[0]==0:
                        predicted_item="dosa"
                        print("dosa")
                elif predicted_classes[0]==1:
                        print("vada")
                        predicted_item="vada"
                else:
                        print("idly")
                        predicted_item="idly"
                print(target_image_pred_probs.max())

                plt.figure()
                plt.imshow(display)
                plt.title(f"Pred: {predicted_item}: {target_image_pred_probs.max() } ")
                plt.axis("off")
                plt.show()

                return 
location="start"
while(location !="stop"):
        location=str(input("give me the file location or type stop  "))
        if location=="stop":
                break 
        deploy_and_test(location)

                










