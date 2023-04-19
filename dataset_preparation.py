"imoort required file "
import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects
#import efficientnet.tfkeras as efn
from tqdm import tqdm
import fnmatch

'Access dataset'

IMG_ROOT = './cataractdataset/dataset/'
IMG_DIR = [IMG_ROOT+'1_normal', 
           IMG_ROOT+'2_cataract', 
           IMG_ROOT+'2_glaucoma', 
           IMG_ROOT+'3_retina_disease']
#ocular -disease-recognition dataset
OCU_IMG_ROOT = './Ocular disease Recognition/preprocessed_images/'
ocu_df = pd.read_csv('./Ocular disease Recognition/full_df.csv')

#Cataract dataset Process

cat_df = pd.DataFrame(0, 
                  columns=['paths', 
                           'cataract'],
                  index=range(601))


filepaths = []
for root, dirnames, filenames in os.walk(IMG_ROOT):
    for filename in fnmatch.filter(filenames, '*.png'):
        filepaths.append(os.path.join(root, filename))
for i, filepath in enumerate(filepaths):
    filepath = os.path.split(filepath)
    cat_df.iloc[i, 0] = filepath[0] + '/' + filepath[1]
    if filepath[0] == IMG_DIR[0]:    # normal
        cat_df.iloc[i, 1] = 0
    elif filepath[0] == IMG_DIR[1]:  # cataract
        cat_df.iloc[i, 1] = 1
    elif filepath[0] == IMG_DIR[2]:  # glaucoma
        cat_df.iloc[i, 1] = 2
    elif filepath[0] == IMG_DIR[3]:  # retine_disease
        cat_df.iloc[i, 1] = 3

        'take only cataract and normal images from cataract dataset'
        
cat_df = cat_df.query('0 <= cataract < 2')

'Process of Occular Disease dataset'

def has_cataract_mentioned(text):
    if 'cataract' in text:
        return 1
    elif 'normal' in text:
        return 0
    else:
        return 2
ocu_df['left_eye_cataract'] = ocu_df['Left-Diagnostic Keywords'].apply(lambda x: has_cataract_mentioned(x))
ocu_df['right_eye_cataract'] = ocu_df['Right-Diagnostic Keywords'].apply(lambda x: has_cataract_mentioned(x))

le_df = ocu_df.loc[:, ['Left-Fundus', 'left_eye_cataract']]\
        .rename(columns={'left_eye_cataract':'cataract'})
le_df['paths'] = OCU_IMG_ROOT + le_df['Left-Fundus']

re_df = ocu_df.loc[:, ['Right-Fundus', 'right_eye_cataract']]\
.rename(columns={'right_eye_cataract':'cataract'})
re_df['paths'] = OCU_IMG_ROOT + re_df['Right-Fundus']

'keep only cataract and normal dataset'
le_df= le_df.query('0 <= cataract < 2')
re_df= re_df.query('0 <= cataract < 2')

'downsample for removing class imbalance'

def downsample(df):
    df = pd.concat([df.query('cataract==1'),
        df.query('cataract==0').sample(sum(df['cataract']), 
                                       random_state=SEED)
    ])
    return df
le_df = downsample(le_df)
re_df = downsample(re_df)

re_df = re_df.drop('Right-Fundus', axis=1)
le_df = le_df.drop('Left-Fundus', axis=1)

'concate left and right fundus images'
ocu_df = pd.concat([le_df, re_df], axis=0)

' concate two independent dataset'
df = pd.concat([cat_df, ocu_df], ignore_index=True)

'save the images '
destinationFolder = "./CombinedCataract/Dataset/"
keys = ["Normal", "Cataract"]
for key in keys:
    if not os.path.exists(destinationFolder+'/'+ key):
        os.makedirs(destinationFolder+'/'+key)
  
  def save_img(data_f):
    n=0
    c=0
    for i, path in enumerate(data_f['paths']):
        img= cv2.imread(path)
        
        lebel= data_f[(data_f['paths']==path)]['cataract']
        
        file_name_no_extension = os.path.splitext(filename)[0]
        #print(file_name_no_extension)
        if lebel[i]== 0:
            cv2.imwrite(destinationFolder+keys[0]+'/'+str(n)+".jpg",img)
            n+=1
            #print(i)
        else:
            cv2.imwrite(destinationFolder+keys[1]+'/'+str(c)+".jpg",img)
            c+=1
            #print(i)
  save_img(df)
