from tensorflow.keras.layers import  Conv2D, BatchNormalization, Activation, DepthwiseConv2D,  Concatenate, Add
from tensorflow.keras.models import Model

from tensorflow.keras.activations import swish
from tensorflow_addons.layers import StochasticDepth
from tensorflow.keras.regularizers import L2
import numpy as np
 
from Layer.Activation import HardSwish
from Layer.Attention import SE_block, CBAM_block, CA_block


def CSP_block(x, exp_size, filters, kernel_size, strides=1, padding='same', activation='HS',
              SE_CBAM_CA="SE", name='CSP', dropout=1, initializer='random_uniform'):
    #--------Part 1--------------------
    # part_1 = Conv2D(filters//2, kernel_size=1, padding=padding)(x)
    part_1 = MBConv(x,
                    exp_size=exp_size[0]//2,
                    filters=filters//2, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    activation=activation,
                    SE_CBAM_CA=SE_CBAM_CA,
                    padding=padding,
                    name=name+"_MBblock{0}".format(1),
                    dropout=dropout,
                    initializer=initializer)
    
    for i in range(1,len(exp_size)):
        part_1 = MBConv(part_1,
                        exp_size=exp_size[i]//2,
                        filters=filters//2,
                        kernel_size=kernel_size,
                        strides=1,
                        activation=activation,
                        SE_CBAM_CA=SE_CBAM_CA,
                        padding=padding,
                        name=name+"_MBblock{0}".format(i+1),
                        dropout=dropout,
                        initializer=initializer)
    
    #--------Part 2--------------------
    # if r==False:
    if strides == 2:
        # x = MaxPool2D()(x)
        x = DepthwiseConv2D(kernel_size=2, strides=(strides,strides), padding=padding,
                            name=name+"_zoom", depthwise_regularizer=L2(0.00005),
                            kernel_initializer=initializer)(x)
        x = BatchNormalization(name=name+"_zoom_bn")(x)
        if Activation=="HS":
            x = HardSwish(name=name+"_zoom_hardswish")(x)
        elif activation == 'RE':
            x = Activation('relu', name=name+'_zoom_relu')(x)
        elif activation == 'SW':
            x = swish(x, name=name+'_zoom_swish')
            
    part_2 = Conv2D(filters//2, 1, padding='same', name=name+"_conv1", kernel_regularizer=L2(0.00005),
                    kernel_initializer=initializer)(x)
    part_2 = BatchNormalization(name=name+"_bn1")(part_2)
    if Activation=="HS":
        part_2 = HardSwish(name=name+"_hardswish1")(part_2)
    elif activation == 'RE':
        part_2 = Activation('relu', name=name+'_relu3')(part_2)
    elif activation == 'SW':
        part_2 = swish(part_2)
    
    #--------output--------------------
    x = Concatenate(name=name+"_concatenate")([part_1, part_2])
    return x
    
def MBConv(x,exp_size, filters, kernel_size, strides=1, padding='same', activation='HS',
           SE_CBAM_CA="SE", name='',dropout=1, initializer='random_uniform'):
    _,_,_,c = np.shape(x)
    #----------------------part_1--------------------------------------
    # Conv1x1
    part_1 = Conv2D(filters=exp_size, kernel_size=1, strides=(1,1), padding='same',
                    name=name+'_conv1', kernel_regularizer=L2(0.00005),
                    kernel_initializer=initializer)(x)
    part_1 = BatchNormalization(name=name + '_bn1')(part_1)
    if activation == 'HS':
        part_1 = HardSwish(name=name+'_hardswish1')(part_1)
    elif activation == 'RE':
        part_1 = Activation('relu', name=name+'_relu1')(part_1)
    elif activation == 'SW':
        part_1 = swish(part_1)
    
    # depthwise conv3x3
    part_1 = DepthwiseConv2D(kernel_size=kernel_size, strides=(strides,strides), padding=padding,
                             name=name+'_dw1', depthwise_regularizer=L2(0.00005),
                             kernel_initializer=initializer)(part_1)
    part_1 = BatchNormalization(name=name+"_bn2")(part_1)
    if activation == 'HS':
        part_1 = HardSwish(name=name+'_hardswish2')(part_1)
    elif activation == 'RE':
        part_1 = Activation('relu', name=name+'_relu2')(part_1)
    elif activation == 'SW':
        part_1 = swish(part_1)
    
    # SE block
    if SE_CBAM_CA==None:
        pass
    elif SE_CBAM_CA == 'SE':
        part_1 = SE_block(part_1,name=name)
    elif SE_CBAM_CA == 'CBAM':
        part_1 = CBAM_block(part_1,name=name)
    elif SE_CBAM_CA == 'CA':
        part_1 = CA_block(part_1,name=name)     
    
    # Conv1x1
    part_1 = Conv2D(filters=filters, kernel_size=1, strides=(1,1), padding='same',
                    name=name+"_conv2", kernel_regularizer=L2(0.00005),
                    kernel_initializer=initializer)(part_1)
    part_1 = BatchNormalization(name=name+"_bn3")(part_1)
    if activation == 'HS':
        part_1 = HardSwish(name=name+'_hardswish3')(part_1)
    elif activation == 'RE':
        part_1 = Activation('relu', name=name+'_relu3')(part_1)
    elif activation == 'SW':
        part_1 = swish(part_1)
    
    # --------------------part_2_resblock------------------------------
    # if r:
    if c == filters and strides==1:
        if dropout != None:
            x = StochasticDepth(dropout, name=name+"_stochastic_depth")([x, part_1])
        else:
            x = Add(name=name+"_add")([x, part_1])
        return x
    return part_1

