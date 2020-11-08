#!/usr/bin/env python
# coding: utf-8

# # Reading data

# In[1]:


# image processing
import numpy as np
#import imageio
import tifffile as tiff
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
import math
import pickle

#modeling 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard

mpl.rcParams['figure.figsize'] = (10,10)


# In[2]:


data_path = '/home/jliu0604/AML/satelite_image/data'
PATCH_SIZE = 160


# In[ ]:


# test reading image
img = tiff.imread(data_path + '/gt_mband/01.tif')


# In[ ]:


#img.shape


# In[ ]:


#plt.imshow(img[1,:,:])


# In[ ]:


#tiff.imshow(img[(3,2,1),:,:])


# In[ ]:





# In[ ]:


# superimpose mask on to the satelite images
st_img = tiff.imread(data_path + '/mband/01.tif')
mask = tiff.imread(data_path + '/gt_mband/01.tif')


def superimpose_stlite_mask(st_img, mask, color = (10,0,0)):
        #normalize image and select only three channels to display
        st_normed = 255.0 * st_img / st_img.max()
        # create color mask using RGB channles
        colored_mask = np.stack([mask*color[0], mask*color[1], mask*color[2]])
        # combine the colored_mask and st_img together
        combined = (st_normed + colored_mask).clip(0,255).astype(np.uint8)
        return combined


# In[ ]:


#combined = superimpose_stlite_mask(st_img[(4,2,1),:,:], mask[0,:,:])
#tiff.imshow(combined)


# In[ ]:





# # Normalize the image
# Normalize image to make each pixel within the range of [-1,1]
# 
# Since we need to compare the input image 𝐼 to the output image 𝐼ˆ, it should be readily possible to enforce the pixel values of 𝐼ˆ into a simple, known, hard range. Using sigmoid produces values in [0,1], while using tanh does so in [−1,1]. However, it is often thought that that tanh is better than sigmoid; e.g.,
# 
# https://stats.stackexchange.com/questions/142348/tanh-vs-sigmoid-in-neural-net
# 
# https://stats.stackexchange.com/questions/330559/why-is-tanh-almost-always-better-than-sigmoid-as-an-activation-function/369538
# 
# https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function
# 
# In other words, for cases where the output must match the input, using [−1,1] may be a better choice. Furthermore, though not "standardized", the range [−1,1] is still zero-centered (unlike [0,1]), which is easier for the network to learn to standardize (though I suspect this matters only rather early in training). https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1

# In[ ]:


def normalize_images(img):
    normalized = 2 * (img - img.min()) / (img.max() - img.min()) - 1
    return normalized


# In[ ]:


#st_img.max()


# In[ ]:


#normalize_images(st_img).max()


# In[ ]:


#st_img.shape


# 

# # Creating extented images
# Since it's difficult for CNN to train on the image edges, creating extended edges will help on training. We will fill the extended edges using mirror images. So far, just extended the right and bottom part, but this can be further improved by extend all four edges.

# In[ ]:


# creat extented x and mask
# def create_extended_imgs(x, mask, PATCH_SIZE = PATCH_SIZE):
#     img_height = x.shape[1]
#     img_width = x.shape[2]
#     n_channels = x.shape[0]
#     n_classes = mask.shape[0]
#     # make extended image with mirror filler
#     npatches_vertical = math.ceil(img_height / PATCH_SIZE)
#     npatches_horizontal = math.ceil(img_width / PATCH_SIZE)
#     ext_x = np.zeros(shape = (n_channels, npatches_vertical*PATCH_SIZE, npatches_horizontal*PATCH_SIZE), dtype = np.float32)
#     ext_m = np.zeros(shape = (n_classes, npatches_vertical*PATCH_SIZE, npatches_horizontal*PATCH_SIZE), dtype = np.float32)
#     #print(ext_x.shape)
#     # fill extended image with mirror filler
#     ext_x[:,:img_height, :img_width] = x
#     ext_m[:,:img_height, :img_width] = mask
#     for i in range(img_height, ext_x.shape[1]):
#         #print(x[:,2*img_height - i -1, :])
#         #can't use ext_x[:,i,:] = x[:,2*img_height - i -1, :], since x, ext_x shape is different
#         ext_x[:,i,:] = ext_x[:,2*img_height - i -1, :]
#         ext_m[:,i,:] = ext_m[:,2*img_height - i -1, :]
#     for i in range(img_width, ext_x.shape[2]):
#         ext_x[:,:,i] = ext_x[:,:,2*img_width -i -1]
#         ext_m[:,:,i] = ext_m[:,:,2*img_width -i -1]

#     return ext_x, ext_m


# In[ ]:





# In[ ]:


# creat extented x and mask
def create_extended_imgs(x, mask = None, PATCH_SIZE = PATCH_SIZE):
    img_height = x.shape[1]
    img_width = x.shape[2]
    n_channels = x.shape[0]
  
    # make extended image with mirror filler
    npatches_vertical = math.ceil(img_height / PATCH_SIZE)
    npatches_horizontal = math.ceil(img_width / PATCH_SIZE)
    ext_x = np.zeros(shape = (n_channels, npatches_vertical*PATCH_SIZE, npatches_horizontal*PATCH_SIZE), dtype = np.float32)
  
    #print(ext_x.shape)
    # fill extended image with mirror filler
    ext_x[:,:img_height, :img_width] = x
   
    for i in range(img_height, ext_x.shape[1]):
        #print(x[:,2*img_height - i -1, :])
        #can't use ext_x[:,i,:] = x[:,2*img_height - i -1, :], since x, ext_x shape is different
        ext_x[:,i,:] = ext_x[:,2*img_height - i -1, :]
    for i in range(img_width, ext_x.shape[2]):
        ext_x[:,:,i] = ext_x[:,:,2*img_width -i -1]

    if mask is not None:
        n_classes = mask.shape[0]
        ext_m = np.zeros(shape = (n_classes, npatches_vertical*PATCH_SIZE, npatches_horizontal*PATCH_SIZE), dtype = np.float32)
        ext_m[:,:img_height, :img_width] = mask
        for i in range(img_height, ext_x.shape[1]):
        #print(x[:,2*img_height - i -1, :])
        #can't use ext_x[:,i,:] = x[:,2*img_height - i -1, :], since x, ext_x shape is different
            ext_m[:,i,:] = ext_m[:,2*img_height - i -1, :]
        for i in range(img_width, ext_x.shape[2]):
            ext_m[:,:,i] = ext_m[:,:,2*img_width -i -1]
        return ext_x, ext_m
    
    return ext_x


# In[ ]:


#filled_img, filled_mask = create_extended_imgs(st_img, mask)


# In[ ]:


#tiff.imshow(filled_img[(4,2,1),:,:])


# In[ ]:


#tiff.imshow(filled_mask[(4,2,1),:,:])


# In[ ]:





# # Creating random patches
# Our image is 837*851, it's usually too large for the network to train, so we will create patches. However, one concern with creating pathes is we might lose information on cropped image edges, so will creating mirror images for the edges.
# 
# Since we can absolutely crop images side by side and creating mirror images on the edges, this might also put us in risk of lossing information on edges. So another way to optimize it is to randomly cropping over an image, that way some of the patches will overlap with each other and get trained.

# In[ ]:


def get_random_patches(img, mask, PATCH_SIZE = PATCH_SIZE):
    assert len(img.shape) == 3 and img.shape[1] > PATCH_SIZE
    assert img.shape[2] > PATCH_SIZE 
    assert img.shape[1:3] == mask.shape[1:3]
    
    xc = random.randint(0, img.shape[2] - PATCH_SIZE)
    yc = random.randint(0, img.shape[1] - PATCH_SIZE)
    
    new_img = img[:, yc : (yc + PATCH_SIZE), xc: (xc + PATCH_SIZE)]
    new_mask = mask[:, yc : (yc + PATCH_SIZE), xc: (xc + PATCH_SIZE)]
    
    return new_img, new_mask


# In[ ]:


#patch_img, pathc_mask = get_random_patches(st_img, mask, 160)


# In[ ]:


#img.shape


# In[ ]:


#plt.imshow(patch_img[5,:,:])


# Now we need to generate random patches for the images

# In[ ]:


def get_patches(img, mask, n_patches, PATCH_SIZE):
    xs = list()
    ys = list()
    
    total_patches = 0
    while total_patches < n_patches:
        img_patch, mask_patch = get_random_patches(img, mask, PATCH_SIZE)
        xs.append(img_patch)
        ys.append(mask_patch)
        total_patches += 1
    print('generated {} pacthes images'.format(total_patches))
    return np.array(xs), np.array(ys)


# In[ ]:


#xs, ys = get_patches(st_img,mask,10,160)


# In[ ]:





# In[ ]:


# if we have a list of images, put them into dictionary, random sampling from the list
def get_patches_batch(x_dict, y_dict, n_patches, sz=PATCH_SIZE):
    x = list()
    y = list()
    
    n_imgs = len(x_dict)
    # make sure each image will get sampled
    sub_npatches = n_patches // n_imgs
    left_npatches = n_patches % n_imgs
    
    total_patches = 0
    

    for i in list(x_dict.keys()):
        while total_patches < int(i)*sub_npatches:
            img = x_dict[i]
            mask = y_dict[i]

            img_patch, mask_patch = get_random_patches(img, mask, PATCH_SIZE)
            x.append(img_patch)
            y.append(mask_patch)

            total_patches += 1
    # The rest images will be filled from image 01        
    while total_patches < n_patches:
        img_patch, mask_patch = get_random_patches(x_dict['01'], y_dict['01'], PATCH_SIZE)
        x.append(img_patch)
        y.append(mask_patch)

        
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)


# In[ ]:


#x_train,y_train=get_patches_batch(X_DICT_TRAIN,Y_DICT_TRAIN, 100, 160)


# In[ ]:


#x_train.shape,y_train.shape


# In[ ]:


#for i in range(1,11):
#    plt.subplot(2,5,i)
#    plt.imshow(xs[i-1][4,:,:])


# In[ ]:





# # Creating predictions
# The prediction will also be doing on the extended images, since this will helps on predicting the edges. After that, will crop the extended area. Similarly, this is base on only right and bottom edge mirror extended prediction. Further imporvement can be achieved.

# In[ ]:


def prediction(img, model, PATCH_SIZE = PATCH_SIZE, n_classes = 5):
    ext_x = create_extended_imgs(img, mask = None)
    img_height = img.shape[1]
    img_width = img.shape[2]
    npatches_vertical = int(ext_x.shape[1] / PATCH_SIZE)
    npatches_horizontal = int(ext_x.shape[2] / PATCH_SIZE)
    
    # assemble all patches into one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i*PATCH_SIZE, (i+1)*PATCH_SIZE
            y0, y1 = j*PATCH_SIZE, (j+1)*PATCH_SIZE
            
            patches_list.append(ext_x[:,y0:y1,x0:x1])
    patches_array = np.asarray(patches_list)
    patches_array = patches_array.transpose([0,2,3,1])
    #print(patches_array.shape)
    return patches_array

# predictions:
    #patches_predict = patches_array[:,:,:,(1,2,3,4,5)]
    #print(patches_predict.shape)
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(ext_x.shape[1], ext_x.shape[2], n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        print(k)
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
        y0, y1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
        print(x0,x1,y0,y1)
        prediction[y0:y1, x0:x1, :] = patches_predict[k,:, :, :]
    return prediction[:img_height, :img_width, :]


# In[ ]:


#5%36


# In[ ]:


#model = 1
#ptest = prediction(test_image, model)


# In[ ]:


#filled_img.shape


# In[ ]:


#ptest.shape


# In[ ]:


#tiff.imshow(ptest[:,:,(4,2,1)])


# In[ ]:


#test_image.shape


# In[ ]:


#tiff.imshow(test_image[4,:,:])


# In[ ]:





# # Create the model

# In[ ]:


def model_unet(n_classes = 5, im_size = PATCH_SIZE, n_channels = 8, n_filters_start =4, growth_factor =2):
    n_filters = n_filters_start
    
    # create model using functional API. It is generally recommend to use the functional layer API via Input, (which creates an InputLayer) without directly using InputLayer.
    inputs = Input((im_size, im_size, n_channels))
    conv1 = Conv2D(n_filters, (3,3), padding = 'same', activation = 'relu')(inputs)
    pool1 = MaxPooling2D((2,2))(conv1)
    n_filters *= growth_factor
    conv2 = Conv2D(n_filters, (3,3), padding = 'same', activation = 'relu')(pool1)
    
    n_filters //= growth_factor
    upconv = Conv2DTranspose(n_filters, (2,2), strides = (2,2), padding = 'same')(conv2)
    concat = concatenate([conv1, upconv])
    conv3 = Conv2D(n_filters, (3,3), activation = 'relu', padding = 'same')(concat)
    output = Conv2D(n_classes, (1,1), activation = 'sigmoid')(conv3)
    model = Model(inputs = inputs, outputs = output)
    
    # compiling model
    model.compile(optimizer = Adam(), loss = 'binary_crossentropy')
    
    return model


# In[ ]:





# In[ ]:


model = model_unet()


# In[ ]:


model.summary()


# In[ ]:


#plot_model(model, show_shapes=True)
#plt.imshow(imageio.imread('simple_unet.png'))


# In[ ]:





# In[ ]:





# # Trainging on model

# In[ ]:


img_ids = [str(i).zfill(2) for i in range(1, 25)] 

X_DICT_TRAIN = dict()
Y_DICT_TRAIN = dict()
X_DICT_VALIDATION = dict()
Y_DICT_VALIDATION = dict()

print('Reading images')
for img_id in img_ids:
    img_m = normalize_images(tiff.imread('/home/jliu0604/AML/satelite_image/data/mband/{}.tif'.format(img_id)))
    # use mask to / 255 put it in the range of [0,1], the end result will be multiply wit 255
    mask = tiff.imread('/home/jliu0604/AML/satelite_image/data/gt_mband/{}.tif'.format(img_id)) / 255
    train_xsz = int(3/4 * img_m.shape[1])  # use 75% of image as train and 25% for validation
    X_DICT_TRAIN[img_id] = img_m[:,:train_xsz, :]
    Y_DICT_TRAIN[img_id] = mask[:,:train_xsz, :]
    X_DICT_VALIDATION[img_id] = img_m[:,:train_xsz, :]
    Y_DICT_VALIDATION[img_id] = mask[:,:train_xsz, :]
    print(img_id + ' read')
print('Images were read')


# In[ ]:


test_image = normalize_images(tiff.imread('/home/jliu0604/AML/satelite_image/data/mband/test.tif'))


# In[ ]:


test_image.shape


# In[ ]:


#check if the data has been normalized, this is channel first
X_DICT_TRAIN['01'].max(),X_DICT_TRAIN['01'].min() ,X_DICT_TRAIN['01'].shape


# In[ ]:


# get random patches for train, test
TRAINING_TOTAL = 100
VALIDATION_TOTAL = 50
x_train, y_train = get_patches_batch(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAINING_TOTAL, sz=PATCH_SIZE)
x_val, y_val = get_patches_batch(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VALIDATION_TOTAL, sz=PATCH_SIZE)


# In[ ]:


#x_train.shape, y_train.shape


# In[ ]:


#x_val.shape, y_val.shape


# In[ ]:





# # Train the model

# In[ ]:


# transpose images to make channel last
x_val = x_val.transpose([0,2,3,1])#
x_val.shape
y_val = y_val.transpose([0,2,3,1])
x_train = x_train.transpose([0,2,3,1])
y_train = y_train.transpose([0,2,3,1])


# In[ ]:


#x_train.shape, y_train.shape


# In[ ]:


#x_val.shape, y_val.shape


# In[ ]:


# Now training the model:
N_EPOCHS = 100
BATCH_SIZE = 32
# ask Keras to save best weights (in terms of validation loss) into file:
model_checkpoint = ModelCheckpoint(filepath='weights_simple_unet.hdf5', monitor='val_loss', save_best_only=True)
# ask Keras to log each epoch loss:
csv_logger = CSVLogger('log.csv', append=True, separator=';')
# ask Keras to log info in TensorBoard format:
tensorboard = TensorBoard(log_dir='tensorboard_simple_unet/', write_graph=True, write_images=True)
# Fit:
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          verbose=0, shuffle=True,
          callbacks=[model_checkpoint, csv_logger, tensorboard],
          validation_data=(x_val, y_val))


# In[ ]:


predicted_mask = prediction(test_image, model)


# In[ ]:


with open('predicted_mask.pickle', 'wb') as handle:
    pickle.dump(predicted_mask, handle)

    
with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle)

