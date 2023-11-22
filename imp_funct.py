import cv2 
import numpy as np 
import torch
import matplotlib as plt
from albumentations_1 import train_transforms , train_transforms_1 ,train_transforms_2
def convert_file_name_to_numpy_array(image_df,task=1):
    image_numpy_array=[]
    print(len(image_df))
    total=len(image_df)
    for i in range(total):
        #len_image_df=len(image_df)
        image = cv2.imread(image_df.iloc[i,0])
        if task=="pretrain_last_layer":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (230,230))
        elif task=="Vit":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224,224))

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (32,32))
        image=image.astype("float32")
        image_numpy_array.append(image)
    image_numpy_array=np.array(image_numpy_array)
    return image_numpy_array

def deploy_and_test(image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display=image
        device_1=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image=image.astype("float32")
        image=image/255
        path_to_model="dataset-3_model_29.pt"
        trained_model=torch.load(path_to_model,map_location=torch.device(device_1))

        augmenatations= train_transforms(image=image)
        image=augmenatations["image"]
        with torch.no_grad():
                trained_model.eval()
                image=torch.from_numpy(image)
                image=torch.unsqueeze(image,dim=0)
                image=image.reshape(1,3,32,32)
                print(type(image))

                #print(torch.unsqueeze(image,0).shape)
                image = image.to(device_1)
                outputs = trained_model(image)
                target_image_pred_probs = torch.softmax(outputs, dim=1)
                 #9. Convert prediction probabilities -> prediction labels
                predicted_classes = torch.argmax(target_image_pred_probs, dim=1)                
                print(predicted_classes)
                predicted_item=""
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

                
                return f"Pred: {predicted_item}: {target_image_pred_probs.max()}  Dosa , vada , idly target_image_pred_probs "
                
 
# location="start"
# while(location !="stop"):
#         location=str(input("give me the file location or type stop  "))
#         if location=="stop":
#                 break 
#         deploy_and_test(location)






