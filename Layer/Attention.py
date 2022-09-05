from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, \
    Reshape, Concatenate, Multiply, Add, GlobalAveragePooling2D,GlobalMaxPooling2D, Permute
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.regularizers import L2

from tensorflow import split, reduce_mean, reduce_max,reshape

from Layer.Activation import HardSigmoid, HardSwish
    


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