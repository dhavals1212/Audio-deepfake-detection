#Import necessary libraries.

import datetime
from time import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

#Add path of the dataset to create train, val, test folders.
dataset_path = "dataset/"

#Create day Training and night Training paths.
day_train_path = "dataset/Annotations/Annotations/dayTrain/"
night_train_path = "dataset/Annotations/Annotations/nightTrain/"

#Create directories of train, val and test, in each of them create images folder, in train and val create labels folder.
os.makedirs(os.path.join(dataset_path, "train/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "test/images"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "val/labels"), exist_ok=True)

#Created empty sets to populate after concatenation of datas together.
day_train = []
night_train = []

#For each folder of dayClip, add all frameAnnotationsBOX.csv in a single df.
for clip in tqdm(sorted(os.listdir(day_train_path))):
    if 'dayClip' not in clip:
        continue
    df = pd.read_csv(os.path.join(day_train_path, clip, 'frameAnnotationsBOX.csv'), sep=';')
    day_train.append(df)

#Concatenate all the data together and add a feature called isNight which is zero.
dayTrain_df = pd.concat(day_train, axis=0)
dayTrain_df['isNight']=0

for clip in tqdm(sorted(os.listdir(night_train_path))):
    if 'nightClip' not in clip:
        continue
    df = pd.read_csv(os.path.join(night_train_path, clip, 'frameAnnotationsBOX.csv'), sep=';')
    night_train.append(df)

nightTrain_df = pd.concat(night_train, axis=0)
nightTrain_df['isNight']=0

#Concatenate every data altogether in single df.
df = pd.concat([dayTrain_df, nightTrain_df], axis=0)

#Check if the columns in comparison in the df are actually duplicates.
np.all(df['Origin file'] == df['Origin track']), np.all(df['Origin frame number'] == df['Origin track frame number'])

#Dropping unwanted columns, not useful to our data in df.
df = df.drop(['Origin file', 'Origin track', 'Origin track frame number'], axis=1)

#Data in the csv contains filenames which are inaccurate, so changing that
def changeFilename(x):
    filename = x.Filename
    isNight = x.isNight
    splitted = filename.split('/')
    clip = splitted[-1].split('--')[0]
    if isNight:
        return os.path.join(dataset_path, f'nightTrain/nightTrain/{clip}/frames/{splitted[-1]}')
    else:
        return os.path.join(dataset_path, f'dayTrain/dayTrain/{clip}/frames/{splitted[-1]}')

df['Filename'] = df.apply(changeFilename, axis=1)

#Find unique data
df['Annotation tag'].unique()

#Changing the annotations to only 3 classes from 6 classes.
label_to_id = {'go':1, 'warning':2, 'stop':3}
id_to_label = {v:k for k, v in label_to_id.items()}

def changeAnnotation(x):
    if 'go' in x['Annotation tag']:
        return label_to_id['go']
    elif 'warning' in x['Annotation tag']:
        return label_to_id['warning']
    elif 'stop' in x['Annotation tag']:
        return label_to_id['stop']

df['Annotation tag'] = df.apply(changeAnnotation, axis=1)

annotaion_tags = df['Annotation tag'].unique()
annotaion_tags

#Changing column names
df.columns = ['image_id', 'label', 'x_min', 'y_min', 'x_max', 'y_max', 'frame', 'isNight']

print("Number of Unique images: ", df.image_id.nunique(), '/', df.shape[0])

#Checking the results
fix, ax = plt.subplots(len(annotaion_tags), 1, figsize=(15, 10*len(annotaion_tags)))

for i, tag in enumerate(annotaion_tags):
    sample = df[df['label']==tag].sample(1)
    bbox = sample[['x_min', 'y_min', 'x_max', 'y_max']].values[0]

    image_path = sample.image_id.values[0]
    #print("Image file path:", sample.image_id.values[0])
    print("Image file path:", image_path)

    image = cv2.imread(sample.image_id.values[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (220, 0, 0), 2)

    ax[i].set_title(id_to_label[tag])
    ax[i].set_axis_off()
    ax[i].imshow(image)