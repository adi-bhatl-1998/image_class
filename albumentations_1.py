
import matplotlib.pyplot as plt 
import numpy as np
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# image = cv2.imread("dataset/dosa/Image_1_.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_list=[image]

train_transforms=A.Compose(
    [
        A.Resize(width=32,height=32),
        A.RandomCrop(width=32,height=32),
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

train_transforms_1=A.Compose(
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

train_transforms_2=A.Compose(
    [
        A.Resize(width=224,height=224),
        A.RandomCrop(width=224,height=224),
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
#img = imread(image_path, as_gray=True)
 # img = imread(image_path, as_gray=True)
     # normalizing the pixel values
def albumenataions(image_df,task=1,operation="null"):
    image_numpy_array=[]
    image_value_array=[]
    total = len(image_df)
    for i in range(total):
        #len_image_df=len(image_df)
        image = cv2.imread(image_df.iloc[i,0])
        if task=="pretrain_last_layer" or task=="Vit":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=image.astype("float32")
        image=image/255
        if operation=="prediction":
            return 

        for j in range(6):
            if task=="pretrain_last_layer":
                augmenatations= train_transforms_1(image=image)
            elif task=="Vit":
                augmenatations= train_transforms_2(image=image)
            else:
                augmenatations= train_transforms(image=image)
                
            augmented_img=augmenatations["image"]
            image_numpy_array.append(augmented_img)
            image_value_array.append(image_df.iloc[i,1])
            #image_df.loc[len(image_df.index)]=[i[0]+f"_{j}",i[0,1]] 
        
    image_numpy_array=np.array(image_numpy_array)
    
    return image_numpy_array, image_value_array

    



