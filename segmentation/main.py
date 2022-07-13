import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
import cv2
from matplotlib import image
from skimage import color
from skimage import io

seed = 42
np.random.seed = seed


Img_Width=128
Img_Height=128
Img_Channel = 3

Train_Path = 'ISBI2016_ISIC_Part3B_Training_Data/ISBI2016_ISIC_Part3B_Training_Data_1/'
Test_Path = 'ISBI2016_ISIC_Part3B_Test_Data/ISBI2016_ISIC_Part3B_Test_Data_1/'


images = os.listdir(Train_Path)
img_train_rgb = list()                    #all train rgb images string present here
img_train_masks = list()                  #all train masks string present here


for i in images:
    if i.endswith('.jpg'):
        img_train_rgb.append(i)
    elif i.endswith('.png'):
        img_train_masks.append(i)


test_images = os.listdir(Test_Path)
img_test_rgb = list()                   #all test images string present here
img_test_masks = list()                 #all test masks string present here


for x in test_images:
    if x.endswith('.jpg'):
        img_test_rgb.append(x)
    elif x.endswith('.png'):
        img_test_masks.append(x)


X_train_rgb = np.zeros((len(img_train_rgb), Img_Height, Img_Width, Img_Channel), dtype=np.uint8)
Y_train_mask = np.zeros((len(img_train_masks), Img_Height, Img_Width, 1), dtype=bool)


print('\nResizing training images \n')
for n,m in tqdm(enumerate(img_train_rgb), total=len(img_train_rgb)):
    train_img = cv2.imread(Train_Path+ img_train_rgb[n])[:, :, :Img_Channel]
    res = resize(train_img, (Img_Height, Img_Width), mode='constant', preserve_range=True)
    X_train_rgb[n] = res

print('\nResizing Train mask\n')
for m,n in tqdm(enumerate(img_train_masks), total=len(img_train_masks)):
    train_mask = cv2.imread(Train_Path + img_train_masks[m])
    train_mask_gr = color.rgb2gray(train_mask)

    train_mask_gray = np.expand_dims(resize(train_mask_gr, (Img_Height, Img_Width), mode='constant', preserve_range=True), axis=-1)
    Y_train_mask[m] = train_mask_gray


X_test_rgb = np.zeros((len(img_test_rgb), Img_Height, Img_Width, Img_Channel), dtype=np.uint8)
Y_test_mask = np.zeros((len(img_test_masks), Img_Height, Img_Width, 1), dtype=bool)
sizes_test_rgb = []
sizes_test_mask = []

print("Resizing test images")
for n,o in tqdm(enumerate(img_test_rgb), total=len(img_test_rgb)):
    test_img = cv2.imread(Test_Path+img_test_rgb[n])[:, :, :Img_Channel]
    sizes_test_rgb.append(test_img)
    res = resize(test_img, (Img_Height, Img_Width), mode='constant', preserve_range=True)
    X_test_rgb[n] = res

print("Resizing Test image done!")

print("Resizing test masks")
for m,p in tqdm(enumerate(img_test_masks), total=len(img_test_masks)):
    test_mask = cv2.imread(Test_Path+img_test_masks[m])
    sizes_test_mask.append(test_mask)
    test_mask_gr = color.rgb2gray(test_mask)
    res = np.expand_dims(resize(test_mask_gr, (Img_Height, Img_Width), mode='constant', preserve_range=True),axis=-1)
    Y_test_mask[m] = res

print("Resizing test mask done!")

#
# image_x = random.randint(0,len(X_train_rgb))
# imshow(X_train_rgb[image_x])
# plt.show()
# imshow(np.squeeze(Y_train_mask[image_x]))
# plt.show()
#

inputs = tf.keras.layers.Input((Img_Width, Img_Height, Img_Channel))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)


c1=tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1=tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)



####
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1],axis=3)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5',verbose=1, save_best_only=True)

callback = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(X_train_rgb, Y_train_mask, validation_split=0.1, batch_size=16, epochs=25, callbacks=callback)

idx = random.randint(0,len(X_train_rgb))
preds_train = model.predict(X_train_rgb[:int(X_train_rgb.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train_rgb[int(X_train_rgb.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test_rgb, verbose=1)

preds_train_t = (preds_train>0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test>0.5).astype(np.uint8)

# ix = random.randint(0, len(preds_train_t))
# imshow(X_train_rgb[ix])
# plt.show()
# imshow(np.squeeze(Y_train_mask[ix]))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()
#
# ix = random.randint(0, len(preds_val_t))
# imshow(X_train_rgb[int(X_train_rgb.shape[0]):][ix])
# plt.show()
# imshow(np.squeeze(Y_train_mask[int(Y_train_mask.shape[0]*0.9):][ix]))
# plt.show()
# imshow(np.squeeze(preds_val_t[ix]))
# plt.show()
#
#



