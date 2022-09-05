class SE_block(tensorflow.keras.layers.Layer):
    def __init__(self, r=2, **kwargs):
        super(SE_block,self).__init__(**kwargs)
        self.gap1 = GlobalAveragePooling2D()
        C = x_attention.shape.as_list()[1]
        self.ds1 = Dense(C/r,activation='relu')
        self.bn1 = BatchNormalization()
        self.ds2 = Dense(C)
        self.bn2 = BatchNormalization()
        self.activation1 = HardSigmoid()
        self.reshape = Reshape(target_shape=(-1,1,1,C))
        self.add = tf.keras.layers.Multiply()

    def call(self,inputs):
        x = self.gap1(inputs)
        x = self.ds1(x)
        x = self.bn1(x)
        x = self.ds2(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.reshape(x)
        outputs = self.add(x)

        return outputs

class CBAM_block(tensorflow.keras.layers.Layer):
    def __init__(self,r=2, mlp_name=[], **kwargs):
        super(CBAM_block,self).__init__(**kwargs)
        
        # 獲取Channel個數
        C = x.shape[3]
        # -----------------------------------Channel Attention Module-----------------------------------------------------------
        self.GMP_x = GlobalMaxPooling2D()
        self.GAP_x = GlobalAveragePooling2D()
        
        self.shared_mlp_down = Dense(C//r, activation='relu', name='Shared_MLP_Down' + str(mlp_name))
        GMP_x = shared_mlp_down(GMP_x)
        GAP_x = shared_mlp_down(GAP_x)
        self.shared_mlp_up = Dense(C, activation='sigmoid', name='Shared_MLP_Up' + str(mlp_name))
        GMP_x = shared_mlp_up(GMP_x)
        GAP_x = shared_mlp_up(GAP_x)
        
        self.add = tf.keras.layers.add([GMP_x, GAP_x])
        self.activation = Activation('sigmoid')
        
        # -----------------------------------channel_attention與x相乘-----------------------------------------------------------
        self.reshape = Reshape(target_shape=(-1,1,1,C))
        self.mul1 = tf.keras.layers.Multiply()
        
        # -----------------------------------Spatial Attention Module-----------------------------------------------------------
        self.MP_y = tf.reduce_mean(x, axis=3, keepdims=True)
        self.AP_y = tf.reduce_max(x, axis=3, keepdims=True)
        
        self.concat = tf.keras.layers.Concatenate()([MP_y, AP_y])
        # -----------------------------------channel pooling--------------------------------------------------------------------
        self.channel_pooling = Conv2D(1, 7, padding='same', activation='sigmoid')(y)
        
        # x與y相乘
        self.mul2 = tf.keras.layers.Multiply()([x, y])

        def call(self,inputs):
            x1 = self.GMP_x(inputs)
            x2 = self.GAP_x(inputs)

            x1 = self.shared_mlp_down(x1)
            x2 = self.shared_mlp_down(x2)

            x1 = self.shared_mlp_up(x1)
            x2 = self.shared_mlp_up(x2)

            x = self.add([x1,x2])
            x = self.activation(x)

            x = self.reshape(x)
            x = self.mul1([inputs,x])

            y1 = self.MP_y(x)
            y2 = self.AP_y






class MBblock(tensorflow.keras.layers.Layer):
    def __init__(self,name , x, exp_size, filters, kernel_size, strides=(1,1), padding='same', activation='HS', r=True, attention_mode=True,**kwargs):
        super(MBblock, self).__init__(name=name, **kwargs)

        # conv 1*1 
        self.conv1 = Conv2D(filters=exp_size, kernel_size=1, strides=(1,1), padding='same')
        self.bn = BatchNormalization()
        if activation == "HS":
            self.activation1 =  HardSigmoid()
        elif activation == "RE":
            self.activation = Activation('relu')
        
        # depthwise conv3x3
        dw1 =  DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding)
        # SE block
        if attention_mode==False:
            pass
        elif self.SE_CBAM_CA == 'SE':
            at = SE_block(part_1)
        elif self.SE_CBAM_CA == 'CBAM':
            part_1 = self.CBAM_block(part_1)
        elif self.SE_CBAM_CA == 'CA':
            part_1 = self.CA_block(part_1)

    def call(self)