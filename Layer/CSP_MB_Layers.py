from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation,\
    DepthwiseConv2D, Reshape, Concatenate, Multiply, Add, ReLU, GlobalAveragePooling2D,GlobalMaxPooling2D, Permute
from tensorflow.keras.models import Model
from tensorflow import split, reduce_mean, reduce_max,reshape
from tensorflow.keras import layers
from tensorflow.keras.activations import swish
from tensorflow_addons.layers import StochasticDepth
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.regularizers import L2
import numpy as np
 
from Layer.Activation import HardSigmoid, HardSwish


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

def SE_block(x,r=4,name=''):
    x_attention = GlobalAveragePooling2D(name=name+"_SE_GAP")(x)
    C = x_attention.shape.as_list()[1]
    x_attention = Dense(C/r, name=name+"_SE_Dense1", kernel_regularizer=L2(0.00005))(x_attention)
    x_attention = BatchNormalization(name=name+"_SE_bn1")(x_attention)
    x_attention = HardSwish(name=name+"_SE_hardswish")(x_attention)
    x_attention = Dense(C, name=name+"_SE_Dense2", kernel_regularizer=L2(0.00005))(x_attention)
    x_attention = BatchNormalization(name=name+"_SE_bn2")(x_attention)
    x_attention = HardSigmoid(name=name+"_SE_hardsigmoid")(x_attention)
    x_attention = reshape(x_attention,shape=(-1,1,1,C), name=name+"_SE_Reshape")
    # x_attention = Reshape(target_shape=(1,1,C), name=name+"_SE_Reshape")(x_attention)
    x = Multiply(name=name+"_SE_Multiply")([x, x_attention])
    return x

def CBAM_block(x , r=2, name=''):
    # 獲取Channel個數
    C = x.shape[3] 
    # -----------------------------------Channel Attention Module-----------------------------------------------------------
    GMP_x = GlobalMaxPooling2D(name = name + '_CBAM_GMP')(x)
    GAP_x = GlobalAveragePooling2D(name = name + '_CBAM_GAP')(x)
    
    shared_mlp_down = Dense(C//r, name=name + '_CBAM_Shared_MLP_Down')
    GMP_x = shared_mlp_down(GMP_x)
    GMP_x = HardSigmoid(name=name+"_CBAM_GMP_hardsigmoid1")(GMP_x)
    GAP_x = shared_mlp_down(GAP_x)
    GAP_x = HardSigmoid(name=name+"_CBAM_GAP_hardsigmoid2")(GAP_x)
    
    shared_mlp_up = Dense(C, name=name + '_CBAMShared_MLP_Up')
    GMP_x = shared_mlp_up(GMP_x)
    GMP_x = HardSwish(name=name+"_CBAM_GMP_hardswish1")(GMP_x)
    GAP_x = shared_mlp_up(GAP_x)
    GAP_x = HardSwish(name=name+"_CBAM_GAP_hardswish2")(GAP_x)
    
    channel_attention = Add()([GMP_x, GAP_x])
    channel_attention = HardSigmoid(name=name+"_CBAM_ADD_hardsigmoid")(channel_attention)
    
    # -----------------------------------channel_attention與x相乘-----------------------------------------------------------
    channel_attention = Reshape(target_shape=(1,1,C),name=name+"_CBAM_Reshape")(channel_attention)
    x = Multiply(name=name+"_CBAM_Multiply1")([x, channel_attention])
    
    # -----------------------------------Spatial Attention Module-----------------------------------------------------------
    MP_y = reduce_mean(x, axis=3, keepdims=True)
    AP_y = reduce_max(x, axis=3,  keepdims=True)
    y = Concatenate(name=name+"_CBAM_Concatenate")([MP_y, AP_y])
    
    # -----------------------------------channel pooling--------------------------------------------------------------------
    w = x.shape[1] 
    if w % 4 == 0:
        kernel_size = 4
    if w % 7 == 0:
        kernel_size = 7
    y = Conv2D(1, kernel_size, padding='same',name=name+"_CBAM_Conv1")(y)
    y = HardSigmoid(name=name+"_CBAM_hardsigmoid3")(y)
    
    # x與y相乘
    output = Multiply(name=name+"_CBAM_Multiply2")([x, y])
    return output

# Coordinate Attention
def CA_block(x, r=2, name=""):
    n,h,w,c = x.shape
    
    mip = max(8, c // r)
    
    x_h = AdaptiveAveragePooling2D((h,1), name=name+'_Coordinate_AdaptiveAveragePooling_h')(x) # H x 1 x C
    x_h = Permute((3,2,1))(x_h)
    x_w = AdaptiveAveragePooling2D((1,w), name=name+'_Coordinate_AdaptiveAveragePooling_w')(x) # 1 x W x C
    x_w = Permute((3,1,2))(x_w)
    
    xy = Concatenate(name=name+"_Coordinate_Concatenate")([x_h, x_w]) # 1 x (W + H) x C/r
    xy = Permute((2,3,1),name=name+"_Coordinate_Permute")(xy)
    xy = Conv2D(filters=mip, kernel_size=1, strides=(1,1), padding='same', name=name+"_Coordinate_Conv1")(xy)
    xy = BatchNormalization(name=name+"_Coordinate_BatchNormalization")(xy)
    xy = HardSigmoid(name=name+"_Coordinate_HardSigmoid1")(xy)
    
    x_h, x_w = split(xy, num_or_size_splits=2, axis=2, name=name+"_Coordinate_split")
    # x_w = Permute((2,1,3))(x_w)
    
    a_h = Conv2D(filters=c, kernel_size=1, strides=(1,1),padding='same', name=name+"_Coordinate_Conv2")(x_h)
    a_h = HardSigmoid(name=name+"_Coordinate_HardSigmoid2")(a_h)
    a_w = Conv2D(filters=c, kernel_size=1, strides=(1,1),padding='same', name=name+"_Coordinate_Conv3")(x_w)
    a_w = HardSigmoid(name=name+"_Coordinate_HardSigmoid3")(a_w)
    
    x = Multiply(name=name+"_Coordinate_Multiply1")([x, a_h])
    output = Multiply(name=name+"_Coordinate_Multiply2")([x, a_w])
    
    return output  