from Layer.CSP_MB_Layers import *
from tensorflow import math
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, ReLU, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input

from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np


def build_base_model(input_shape=None, SE_CBAM_CA="SE", Specification=None, initializer='random_uniform'):
    Inputs = Input(shape=input_shape)
    # x = Resizing(height=input_shape[1][0],width=input_shape[1][1],interpolation="bicubic")(Inputs)
    x = Inputs

    for Spec in Specification:
        # 0-Operator, 1-exp_size, 2-filters, 3-kernel_size, 4-strides, 5-padding, 6-activation, 7-SE_CBAM_CA, 8-Name
        operator = Spec[0]
        exp_size = Spec[1]
        filters = Spec[2]
        kernel_size = Spec[3]
        strides = Spec[4]
        padding = Spec[5]
        activation = Spec[6]
        # SE_CBAM_CA = Spec[7]
        name = Spec[8]
        drop_out = Spec[9]
        
        if Spec[7] != None:
            Spec[7] = SE_CBAM_CA

        
        if Spec[0] == "Conv":
            x = Conv2D(filters=Spec[2], kernel_size=Spec[3], strides=(Spec[4], Spec[4]), padding=Spec[5], 
                       name=Spec[8], kernel_regularizer=L2(0.00005), kernel_initializer=initializer)(x)
            x = BatchNormalization()(x)
            if Spec[6] =="HS":
                x = HardSwish(name=Spec[8]+"hardswish")(x)
            elif Spec[6] =="RE":
                x = ReLU(name=Spec[8]+"relu")(x)
        elif Spec[0] == 'DWconv':
            x = DepthwiseConv2D(kernel_size=Spec[3], strides=(Spec[4],Spec[4]), padding=Spec[5], activation=Spec[6],
                                name=Spec[8], kernel_initializer=initializer)(x)
        elif Spec[0] == "MBblock":
            x = MBConv(x, exp_size=Spec[1], filters=Spec[2], kernel_size=Spec[3], strides=Spec[4], padding=Spec[5],
                       activation=Spec[6], SE_CBAM_CA=Spec[7], name=Spec[8], dropout=Spec[9], initializer=initializer)
        elif Spec[0] == "CSPblock":
            x = CSP_block(x, exp_size=Spec[1], filters=Spec[2], kernel_size=Spec[3], strides=Spec[4], padding=Spec[5],
                          activation=Spec[6], SE_CBAM_CA=Spec[7], name=Spec[8], dropout=Spec[9], initializer=initializer)
    
    base_model = Model(inputs=Inputs, outputs=x)
    return base_model

def predictions_head(base_model,num_classes,Dropout_rate=0.3):
        x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dropout(rate=Dropout_rate)(x)
        
        #類別數
        x = Dense(num_classes, name="predition_head_classification", kernel_regularizer=L2(0.00005))(x)
        predictions = Activation('softmax', name="predition_softmax")(x)
        return predictions
        # return Model(inputs=base_model.input, outputs=predictions)
    
def setup_to_transfer_learning(base_model, unfrozen=None):
    if unfrozen != None:
        for layer in base_model.layers:
            layer.trainable = False
            for name in unfrozen:
                if name in layer.name:
                    print("layer_name: ",layer.name)
                    layer.trainable = True
                    break

def optimizer_set(LR_mode="Adam", INIT_LR=0.00025, MAX_LR=0.001, steps_per_epoch=None):

    if LR_mode=='CLR' :
        '''
        triangular: scale_fn = lambda x: 1.
        triangular2': scale_fn = lambda x: 1 / (2. ** (x - 1))
        exp_range: scale_fn = lambda x: gamma ** (x)
        Custom Iteration-Policy： fn = lambda x: 1/(5**(x*0.0001))
        Custom Cycle-Policy: scale_fn = lambda x: 0.5*(1+tf.math.sin(x*np.pi/2.)
        '''
        gamma=0.99994
        # 周期性学习率(Cyclical Learning Rate)技术
        clr = CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                   maximal_learning_rate=MAX_LR,
                                   scale_fn=lambda x: 0.5*(1+math.sin(x*np.pi/2.)),
                                   step_size=2 * steps_per_epoch,
                                   # scale_mode='iterations'
                                   )
        return Adam(clr)
    
    elif LR_mode=="Adam" or LR_mode=="adam":
        return Adam()
    elif LR_mode=="RMSprop":
        return RMSprop(1e-4)
    
    # elif LR_mode=="SGD":
    #     lrate = LearningRateScheduler(step_decay)
    #     sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
    #     return sgd
        
# def step_decay(epoch):
# 	initial_lrate = 0.1
# 	drop = 0.5
# 	epochs_drop = 10.0
# 	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
# 	return lrate       


def build_model(base_model,predictions):
    return Model(inputs=base_model.input, outputs=predictions)
                    