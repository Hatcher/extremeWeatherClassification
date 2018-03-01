from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.models import models

def createGoogLeNet():
    # Our shape is 3 for the RGB, 224 for number of pixels in the width, 224 for the number of pixels in the height.
    input = Input(shape=(3, 224, 224) )
    # First convolution layer of googLeNet has a patch size of 7x7 and a stride of 2 pixels.
    # Has an output size of 64, a filter of height 7px, and a stride of 2pixels. These are specified in the first three parameters
    # padding='same', I am going with same as im pretty sure its what the original GoogLeNet used.
    # we are using the relu activation.
    conv1 = Conv2D(64, 7, 2, padding='same', activation='relu')(input)

    # Next we max pool. We pass conv1 as we want to hook up our layers.
    pool1 = MaxPooling2D(3, 2, padding='same', data_format=None)(conv1)

    # Next another 2D convolution
    conv2 = Conv2D(192, 3, 1, padding='same', activation='relu')(pool1)

    # Pooling layer
    pool2 = MaxPooling2D(3, 2, padding='same', data_format=None)(conv2)

    #Now we create the inception layers which are just 2D convolution layers
    inception3A_1 = Conv2D(64, 1, 1, padding='same', activation='relu')(pool2)
    inception3A_3_reduce = Conv2D(96, 1, 1, padding='same', activation='relu')(pool2)
    inception3A_3 = Conv2D(128, 3, 3, padding='same', activation='relu')(inception3A_3_reduce)
    inception3A_5_reduce = Conv2D(16, 1, 1, padding='same', activation='relu')(pool2)
    inception3A_5 = Conv2D(64, 1, 1, padding='same', activation='relu')(inception3A_5_reduce)
    inception3APool = MaxPooling2D(3, 1, padding='same')(pool2)
    inception3APoolProj = Conv2D(32, 1, 1, padding='same', activation='relu')(inception3APool)
    inception3AMerged = concatenate([inception3A_1, inception3A_3, inception3A_5, inception3APoolProj], 1 )

    #Inception 3b
    inception3B_1 = Conv2D(128, 1, 1, padding='same', activation='relu')(inception3AMerged)
    inception3B_3_reduce = Conv2D(128, 1, 1, padding='same', activation='relu')(inception3AMerged)
    inception3B_3 = Conv2D(192, 3, 3, padding='same', activation='relu')(inception3B_3_reduce)
    inception3B_5_reduce = Conv2D(32, 1, 1, padding='same', activation='relu')(inception3AMerged)
    inception3B_5 = Conv2D(96, 1, 1, padding='same', activation='relu')(inception3B_5_reduce)
    inception3BPool = MaxPooling2D(3, 1, padding='same')(inception3AMerged)
    inception3BPoolProj = Conv2D(64, 1, 1, padding='same', activation='relu')(inception3BPool)
    inception3BMerged = concatenate([inception3B_1, inception3B_3, inception3B_5, inception3BPoolProj], 1 )

    #Max pool
    pool3 = MaxPooling2D(3, 2, padding='same', data_format=None)(inception3inception3BMerged)

    #Inception 4A
    inception4A_1 = Conv2D(192, 1, 1, padding='same', activation='relu')(pool3)
    inception4A_3_reduce = Conv2D(96, 1, 1, padding='same', activation='relu')(pool3)
    inception4A_3 = Conv2D(208, 3, 3, padding='same', activation='relu')(inception4A_3_reduce)
    inception4A_5_reduce = Conv2D(16, 1, 1, padding='same', activation='relu')(pool3)
    inception4A_5 = Conv2D(48, 1, 1, padding='same', activation='relu')(inception4A_5_reduce)
    inception4APool = MaxPooling2D(3, 1, padding='same')(pool3)
    inception4APoolProj = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4APool)
    inception4AMerged = concatenate([inception4A_1, inception4A_3, inception4A_5, inception4APoolProj], 1 )

    #Inception 4B
    inception4B_1 = Conv2D(160, 1, 1, padding='same', activation='relu')(inception4AMerged)
    inception4B_3_reduce = Conv2D(112, 1, 1, padding='same', activation='relu')(inception4AMerged)
    inception4B_3 = Conv2D(224, 3, 3, padding='same', activation='relu')(inception4B_3_reduce)
    inception4B_5_reduce = Conv2D(24, 1, 1, padding='same', activation='relu')(inception4AMerged)
    inception4B_5 = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4B_5_reduce)
    inception4BPool = MaxPooling2D(3, 1, padding='same')(inception4AMerged)
    inception4BPoolProj = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4BPool)
    inception4BMerged = concatenate([inception4B_1, inception4B_3, inception4B_5, inception4BPoolProj], 1 )

    #Inception 4C
    inception4C_1 = Conv2D(128, 1, 1, padding='same', activation='relu')(inception4BMerged)
    inception4C_3_reduce = Conv2D(128, 1, 1, padding='same', activation='relu')(inception4BMerged)
    inception4C_3 = Conv2D(256, 3, 3, padding='same', activation='relu')(inception4C_3_reduce)
    inception4C_5_reduce = Conv2D(24, 1, 1, padding='same', activation='relu')(inception4BMerged)
    inception4C_5 = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4C_5_reduce)
    inception4CPool = MaxPooling2D(3, 1, padding='same')(inception4BMerged)
    inception4CPoolProj = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4CPool)
    inception4CMerged = concatenate([inception4C_1, inception4C_3, inception4C_5, inception4CPoolProj], 1 )

    #Inception 4D
    inception4D_1 = Conv2D(112, 1, 1, padding='same', activation='relu')(inception4CMerged)
    inception4D_3_reduce = Conv2D(144, 1, 1, padding='same', activation='relu')(inception4CMerged)
    inception4D_3 = Conv2D(288, 3, 3, padding='same', activation='relu')(inception4D_3_reduce)
    inception4D_5_reduce = Conv2D(32, 1, 1, padding='same', activation='relu')(inception4CMerged)
    inception4D_5 = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4D_5_reduce)
    inception4DPool = MaxPooling2D(3, 1, padding='same')(inception4CMerged)
    inception4DPoolProj = Conv2D(64, 1, 1, padding='same', activation='relu')(inception4DPool)
    inception4DMerged = concatenate([inception4D_1, inception4D_3, inception4D_5, inception4DPoolProj], 1 )

    #Inception 4E
    inception4E_1 = Conv2D(256, 1, 1, padding='same', activation='relu')(inception4DMerged)
    inception4E_3_reduce = Conv2D(160, 1, 1, padding='same', activation='relu')(inception4DMerged)
    inception4E_3 = Conv2D(320, 3, 3, padding='same', activation='relu')(inception4E_3_reduce)
    inception4E_5_reduce = Conv2D(32, 1, 1, padding='same', activation='relu')(inception4DMerged)
    inception4E_5 = Conv2D(128, 1, 1, padding='same', activation='relu')(inception4E_5_reduce)
    inception4EPool = MaxPooling2D(3, 1, padding='same')(inception4DMerged)
    inception4EPoolProj = Conv2D(128, 1, 1, padding='same', activation='relu')(inception4EPool)
    inception4EMerged = concatenate([inception4E_1, inception4E_3, inception4E_5, inception4EPoolProj], 1 )

    #Max pool
    pool4 = MaxPooling2D(3,2, padding='same', data_format=None)(inception4EMerged)

    #Inception5A
    inception5A_1 = Conv2D(256, 1, 1, padding='same', activation='relu')(pool4)
    inception5A_3_reduce = Conv2D(160, 1, 1, padding='same', activation='relu')(pool4)
    inception5A_3 = Conv2D(320, 3, 3, padding='same', activation='relu')(inception5A_3_reduce)
    inception5A_5_reduce = Conv2D(32, 1, 1, padding='same', activation='relu')(pool4)
    inception5A_5 = Conv2D(128, 1, 1, padding='same', activation='relu')(inception5A_5_reduce)
    inception5APool = MaxPooling2D(3, 1, padding='same')(pool4)
    inception5APoolProj = Conv2D(128, 1, 1, padding='same', activation='relu')(inception5APool)
    inception5AMerged = concatenate([inception5A_1, inception5A_3, inception5A_5, inception5APoolProj], 1 )

    #Inception5B
    inception5B_1 = Conv2D(384, 1, 1, padding='same', activation='relu')(inception5AMerged)
    inception5B_3_reduce = Conv2D(192, 1, 1, padding='same', activation='relu')(inception5AMerged)
    inception5B_3 = Conv2D(384, 3, 3, padding='same', activation='relu')(inception5B_3_reduce)
    inception5B_5_reduce = Conv2D(48, 1, 1, padding='same', activation='relu')(inception5AMerged)
    inception5B_5 = Conv2D(128, 1, 1, padding='same', activation='relu')(inception5B_5_reduce)
    inception5BPool = MaxPooling2D(3, 1, padding='same')(inception5AMerged)
    inception5BPoolProj = Conv2D(128, 1, 1, padding='same', activation='relu')(inception5BPool)
    inception5BMerged = concatenate([inception5B_1, inception5B_3, inception5B_5, inception5BPoolProj], 1 )

    #Average pooling layer
    avgPool = AveragePooling2D(pool_size=(7,7), strides=1, padding='same', data_format=None)(inception5BMerged)

    #Dropout
    droppedOut = Dropout(0.40)(avgPool)

    #Linear
    linearLayer = linear()(droppedOut)

    #Softmax
    lastSoftMax = Softmax()(linearLayer)

    #Now we declare the model
    googleNet = Model(input=input, output=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])
