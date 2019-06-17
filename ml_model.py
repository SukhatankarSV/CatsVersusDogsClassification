import cv2
import numpy as np
import os
from random import  shuffle
from tqdm import tqdm #professional looping progreess bar
import numpy as np
import tensorflow as tf




# Training and testing directories where images have been stored
TRAIN_DIR='\\train'
TEST_DIR='\\test'
TEST_DIR2='\\realtest'

#Img size 50*50
IMG_SIZE=50
LR=1e-3 #0.001 Learning rate ...large lr misses out best smaler lr - takes more time

MODEL_NAME='malevsfemale-{}-{}.model'.format(LR,'6conv-basic-video') # when we save model hel to remember us with model name

# 2conv-basic-video is having 2 convolutional layers in the network
#[x ,x]= [catness,dogness] for cat [1 0 ]...for dog [0 1]

# defines whether the provided image is of cat or  a dog by determining [1 0] --catness and [0 1] --dogness


def sv_process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR2)):
        #get path for each image
        path=os.path.join(TEST_DIR2,img)
        #1.png  by splitting we get the image number
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) # read image convert to grayscale and resize to 50*50
        testing_data.append([np.array(img),img_num]) #ad to testing_data list resized grayscale image and image number

    np.save('test_datamf2.npy',testing_data)
    return testing_data



def label_img(img):

    #dog.93.png  ---splitted by (.) -1 would be png -2 would be 93 and -3 would be dog
    word_label=img.split('.')[-3]
    if word_label == 'm':
        return [1,0]
    elif word_label == 'f':
        return [0,1]

#creating the training data
def create_train_data():

    training_data=[]


    for img in tqdm(os.listdir(TRAIN_DIR)): # for every image in training directory

        label=label_img(img) # returns [1,0] -cat and [0,1] -dog
        path=os.path.join(TRAIN_DIR,img) # we get the path of specific image in training directory
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) #read the image from path, convert to grayscale and resize it to IMG_SIZE*IMG_SIZE i.e 50*50
        training_data.append([np.array(img),np.array(label)]) # now add to training data list in the form of (resized grayscale image,label)
    shuffle(training_data)
    np.save('train_datamf.npy',training_data)
    return training_data #shuffle save and return



def process_test_data():
    testing_data=[]

    #for wvery image in test directory
    for img in tqdm(os.listdir(TEST_DIR)):
        #get path for each image
        path=os.path.join(TEST_DIR,img)
        #1.png  by splitting we get the image number
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE)) # read image convert to grayscale and resize to 50*50
        testing_data.append([np.array(img),img_num]) #ad to testing_data list resized grayscale image and image number

    np.save('test_datamf.npy',testing_data)
    return testing_data


train_data=create_train_data()

# if u already have data
# train_data=np.load('train_data.npy')
#----------------------------------------covnet === convo n/w----------------------------------------------------------------------#

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# one conv alyer --linear problem 2 conv layers and more --non linear problems
# we have got 2 conv layers here

# WE CAN USE 6 CON LAYERS HERE just copy paste below code of convnet i.e 85,86 more 2 times ane change 2 to 6 in line 17

convnet = input_data(shape=[None, IMG_SIZE,IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


#new addtion 6 layer convo add 4

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
################

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax') # 2 beccause we have to classify it only as cat or dog
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')


#-----------------------------------------------------------------------------------------------------------------------#

if os.path.exists('{}.meta'.format((MODEL_NAME))): # this means a checkpoint has been saved
    #weights of nw have alreadt been trianed to some extent
    model.load(MODEL_NAME)
    print('Model successfully loaded')

train=train_data[:-428] # trianin data would be all till last 500 smples################changes
test=train_data[-428:] #testing data will be lst 500 samples


# X-featureset y: label

X=np.array([i[0] for i in train]) # this will give the image(grayscaled and resized)
X=X.reshape(-1,IMG_SIZE,IMG_SIZE,1) # training images ready
y=[i[1] for i in train] # training labels i.e cat/dog ready


test_X=np.array([i[0] for i in test]) # this will give the image(grayscaled and resized)
test_X=test_X.reshape(-1,IMG_SIZE,IMG_SIZE,1) #testing images ready
test_y=[i[1] for i in test] #testing labels  i.e imgnum 1,2,3 ... ready


model.fit({'input': X}, {'targets': y}, n_epoch=25, validation_set=({'input': test_X}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



model.save(MODEL_NAME)

# ======================================================================================================================#

import matplotlib.pyplot as plt

# if u don't have this file yet
#test_data=process_test_data() ###############changes

test_data=sv_process_test_data()

# if u already have it

#test_data=np.load('test_data.npy')

fig=plt.figure()

#ierate throough first 12 teste=ing data and plot them on fig and put them up with classification

for num,data in enumerate(test_data): #########changes
    #cat: [1,0] and dog : [0,1]

    img_num=data[1]
    img_data=data[0]

    y=fig.add_subplot(2,8,num+1) # 3 by 4 subplot grid and number itslef is number +1 #########changes
    orig=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    model_out=model.predict([data])[0] # it returnslist we r interested in 0th element only

    if(np.argmax(model_out)==1):
        str_label='Female'
    else:
        str_label='Male'

    y.imshow(orig,cmap='gray')
    plt.title(str_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

with open('submission-file.csv','w') as f:
    f.write('id,label\n')

with open('submission-file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]

        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

        model_out = model.predict([data])[0]

        f.write('{},{}\n'.format(img_num,model_out[1]))
