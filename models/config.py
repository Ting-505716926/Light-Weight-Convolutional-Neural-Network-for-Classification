#Operator, exp_size, filters, kernel_size, strides, padding, activation, SE_CBAM_CA, Name
def MobileNetLarge(Dataset=None):
    if Dataset[:5] == "cifar":
        Specification = [['Conv',   None, 16,  3, 1, "same", "HS", None, "Conv1",  None],
                        ['MBblock', 16  , 16,  3, 1, "same", "RE", None, "Bneck1", 1-0.083*1],
                        ['MBblock', 64  , 24,  3, 1, "same", "RE", None, "Bneck2", 1-0.083*2],
                        ['MBblock', 72  , 24,  3, 1, "same", "RE", None, "Bneck3", 1-0.083*2],
                        ['MBblock', 72  , 40,  5, 2, "same", "RE", "SE", "Bneck4", 1-0.083*3],
                        ['MBblock', 120 , 40,  5, 1, "same", "RE", "SE", "Bneck5", 1-0.083*3],
                        ['MBblock', 120 , 40,  5, 1, "same", "RE", "SE", "Bneck6", 1-0.083*3],
                        ['MBblock', 240 , 80,  3, 2, "same", "HS", None, "Bneck7", 1-0.083*4],
                        ['MBblock', 200 , 80,  3, 1, "same", "HS", None, "Bneck8", 1-0.083*4],
                        ['MBblock', 184 , 80,  3, 1, "same", "HS", None, "Bneck9", 1-0.083*4],
                        ['MBblock', 184 , 80,  3, 1, "same", "HS", None, "Bneck10",1-0.083*4],
                        ['MBblock', 480 , 112, 3, 1, "same", "HS", "SE", "Bneck11",1-0.083*5],
                        ['MBblock', 672 , 112, 3, 1, "same", "HS", "SE", "Bneck12",1-0.083*5],
                        ['MBblock', 672 , 160, 5, 2, "same", "HS", "SE", "Bneck13",1-0.083*6],
                        ['MBblock', 672 , 160, 5, 1, "same", "HS", "SE", "Bneck14",1-0.083*6],
                        ['MBblock', 960 , 160, 5, 1, "same", "HS", "SE", "Bneck15",1-0.083*6],
                        ['Conv',    None, 960, 1, 1, "same", "HS", None, "Conv2",  None],
                        ['Conv',    None, 1280,1, 1, "same", "HS", None, "Conv3",  None],
                        ]
        
    else:
        Specification = [['Conv',   None, 16,  3, 2, "same", "HS", None, "Conv1",  None],
                        ['MBblock', 16  , 16,  3, 1, "same", "RE", None, "Bneck1", 1-0.083*1],
                        ['MBblock', 64  , 24,  3, 2, "same", "RE", None, "Bneck2", 1-0.083*2],
                        ['MBblock', 72  , 24,  3, 1, "same", "RE", None, "Bneck3", 1-0.083*2],
                        ['MBblock', 72  , 40,  5, 2, "same", "RE", "SE", "Bneck4", 1-0.083*3],
                        ['MBblock', 120 , 40,  5, 1, "same", "RE", "SE", "Bneck5", 1-0.083*3],
                        ['MBblock', 120 , 40,  5, 1, "same", "RE", "SE", "Bneck6", 1-0.083*3],
                        ['MBblock', 240 , 80,  3, 2, "same", "HS", None, "Bneck7", 1-0.083*4],
                        ['MBblock', 200 , 80,  3, 1, "same", "HS", None, "Bneck8", 1-0.083*4],
                        ['MBblock', 184 , 80,  3, 1, "same", "HS", None, "Bneck9", 1-0.083*4],
                        ['MBblock', 184 , 80,  3, 1, "same", "HS", None, "Bneck10",1-0.083*4],
                        ['MBblock', 480 , 112, 3, 1, "same", "HS", "SE", "Bneck11",1-0.083*5],
                        ['MBblock', 672 , 112, 3, 1, "same", "HS", "SE", "Bneck12",1-0.083*5],
                        ['MBblock', 672 , 160, 5, 2, "same", "HS", "SE", "Bneck13",1-0.083*6],
                        ['MBblock', 672 , 160, 5, 1, "same", "HS", "SE", "Bneck14",1-0.083*6],
                        ['MBblock', 960 , 160, 5, 1, "same", "HS", "SE", "Bneck15",1-0.083*6],
                        ['Conv',    None, 960, 1, 1, "same", "HS", None, "Conv2",  None],
                        ['Conv',    None, 1280,1, 1, "same", "HS", None, "Conv3",  None],
                        ]
    unfrozen = ["Conv2", "Conv3", "Bneck13", "Bneck14", "Bneck15"]
    return Specification, unfrozen
    
def MobileNetSamll(Dataset=None):
    if Dataset[:5] == "cifar":
        Specification = [['Conv',   None, 16,  3, 1, "same", "HS", None, "Conv1",    None],
                        ['MBblock', 16  , 16,  3, 1, "same", "RE", "SE", "Bneck1",  1-0.01*1],
                        ['MBblock', 72  , 24,  3, 2, "same", "RE", None, "Bneck2",  1-0.01*2],
                        ['MBblock', 88  , 24,  3, 1, "same", "RE", None, "Bneck3",  1-0.01*2],
                        ['MBblock', 96  , 40,  5, 2, "same", "HS", "SE", "Bneck4",  1-0.01*3],
                        ['MBblock', 240 , 40,  5, 1, "same", "HS", "SE", "Bneck5",  1-0.01*3],
                        ['MBblock', 240 , 40,  5, 1, "same", "HS", "SE", "Bneck6",  1-0.01*3],
                        ['MBblock', 120 , 48,  5, 1, "same", "HS", "SE", "Bneck7",  1-0.01*4],
                        ['MBblock', 144 , 48,  5, 1, "same", "HS", "SE", "Bneck8",  1-0.01*4],
                        ['MBblock', 288 , 96,  5, 2, "same", "HS", "SE", "Bneck9",  1-0.01*5],
                        ['MBblock', 576 , 96,  5, 1, "same", "HS", "SE", "Bneck10", 1-0.01*5],
                        ['MBblock', 576 , 96,  5, 1, "same", "HS", "SE", "Bneck11", 1-0.01*5],
                        ['Conv',    None, 576, 1, 1, "same", "HS", None, "Conv2",   None],
                        ['Conv',    None, 1024,1, 1, "same", "HS", None, "Conv3",   None],
                        ]
    else:
        Specification = [['Conv',   None, 16,  3, 2, "same", "HS", None, "Conv1",   None],
                        ['MBblock', 16  , 16,  3, 2, "same", "RE", "SE", "Bneck1",  1-0.01*1],
                        ['MBblock', 72  , 24,  3, 2, "same", "RE", None, "Bneck2",  1-0.01*2],
                        ['MBblock', 88  , 24,  3, 1, "same", "RE", None, "Bneck3",  1-0.01*2],
                        ['MBblock', 96  , 40,  5, 2, "same", "HS", "SE", "Bneck4",  1-0.01*3],
                        ['MBblock', 240 , 40,  5, 1, "same", "HS", "SE", "Bneck5",  1-0.01*3],
                        ['MBblock', 240 , 40,  5, 1, "same", "HS", "SE", "Bneck6",  1-0.01*3],
                        ['MBblock', 120 , 48,  5, 1, "same", "HS", "SE", "Bneck7",  1-0.01*4],
                        ['MBblock', 144 , 48,  5, 1, "same", "HS", "SE", "Bneck8",  1-0.01*4],
                        ['MBblock', 288 , 96,  5, 2, "same", "HS", "SE", "Bneck9",  1-0.01*5],
                        ['MBblock', 576 , 96,  5, 1, "same", "HS", "SE", "Bneck10", 1-0.01*5],
                        ['MBblock', 576 , 96,  5, 1, "same", "HS", "SE", "Bneck11", 1-0.01*5],
                        ['Conv',    None, 576, 1, 1, "same", "HS", None, "Conv2",   None],
                        ['Conv',    None, 1024,1, 1, "same", "HS", None, "Conv3",   None],
                        ]
    unfrozen = ["Conv2", "Conv3", "Bneck9", "Bneck10", "Bneck11"]
    return Specification, unfrozen

def CustomizeSmall(Dataset=None):
    if Dataset[:5] == "cifar":
        Specification = [['Conv',     None         , 16,  3, 1, "same", "HS", None, "Conv1", None],
                         ['CSPblock', [16]         , 16,  3, 1, "same", "RE", "SE", "CSP1",  1-0.01*1],
                         ['CSPblock', [72,88]      , 24,  3, 2, "same", "RE", None, "CSP2",  1-0.01*2],
                         ['CSPblock', [96,240,240] , 40,  5, 2, "same", "HS", "SE", "CSP3",  1-0.01*3],
                         ['CSPblock', [120,144]    , 48,  5, 1, "same", "HS", "SE", "CSP4",  1-0.01*4],
                         ['CSPblock', [288,576,576], 96,  5, 2, "same", "HS", "SE", "CSP5",  1-0.01*5],
                         ['Conv',     None         , 576, 1, 1, "same", "HS", None, "Conv2", None],
                         ['Conv',     None         , 1024,1, 1, "same", "HS", None, "Conv3", None],
                        ]
    else:
        Specification = [['Conv',     None         , 16,  3, 2, "same", "HS", None, "Conv1", None],
                         ['CSPblock', [16]         , 16,  3, 2, "same", "RE", "SE", "CSP1",  1-0.01*1],
                         ['CSPblock', [72,88]      , 24,  3, 2, "same", "RE", None, "CSP2",  1-0.01*2],
                         ['CSPblock', [96,240,240] , 40,  5, 2, "same", "HS", "SE", "CSP3",  1-0.01*3],
                         ['CSPblock', [120,144]    , 48,  5, 1, "same", "HS", "SE", "CSP4",  1-0.01*4],
                         ['CSPblock', [288,576,576], 96,  5, 2, "same", "HS", "SE", "CSP5",  1-0.01*5],
                         ['Conv',     None         , 576, 1, 1, "same", "HS", None, "Conv2", None],
                         ['Conv',     None         , 1024,1, 1, "same", "HS", None, "Conv3", None],
                        ]
    unfrozen = ["Conv2", "Conv3", "CSP3"]
    
    return Specification, unfrozen
 
def CustomizeLarge(Dataset=None):
    if Dataset[:5] == "cifar":
        Specification = [['Conv',     None             , 16,   3, 1, "same", "HS", None, "Conv1", None],
                         ['CSPblock', [16]             , 16,   3, 1, "same", "RE", None, "CSP1" , 1-0.083*1],
                         ['CSPblock', [64,72]          , 24,   3, 1, "same", "RE", None, "CSP2" , 1-0.083*2],
                         ['CSPblock', [72,120,120]     , 40,   5, 2, "same", "HS", "SE", "CSP3" , 1-0.083*3],
                         ['CSPblock', [240,200,184,184], 80,   5, 2, "same", "HS", None, "CSP4" , 1-0.083*4],
                         ['CSPblock', [480,672]        , 112,  5, 1, "same", "HS", "SE", "CSP5" , 1-0.083*5],
                         ['CSPblock', [672,960,960]    , 160,  5, 2, "same", "HS", "SE", "CSP6" , 1-0.083*6],
                         ['Conv',     None             , 960,  1, 1, "same", "HS", None, "Conv2", None],
                         ['Conv',     None             , 1280, 1, 1, "same", "HS", None, "Conv3", None],
                        ]
    else:
        Specification = [# 1/2
                         ['Conv',     None             , 16,   3, 2, "same", "HS", None, "Conv1", None], 
                         ['CSPblock', [16]             , 16,   3, 1, "same", "RE", None, "CSP1" , 1-0.083*1],
                         # 1/4
                         ['CSPblock', [64,72]          , 24,   3, 2, "same", "RE", None, "CSP2" , 1-0.083*2],
                         # 1/8
                         ['CSPblock', [72,120,120]     , 40,   5, 2, "same", "HS", "SE", "CSP3" , 1-0.083*3],
                         # 1/16
                         ['CSPblock', [240,200,184,184], 80,   5, 2, "same", "HS", None, "CSP4" , 1-0.083*4],
                         ['CSPblock', [480,672]        , 112,  5, 1, "same", "HS", "SE", "CSP5" , 1-0.083*5],
                         # 1/32
                         ['CSPblock', [672,960,960]    , 160,  5, 2, "same", "HS", "SE", "CSP6" , 1-0.083*6],
                         ['Conv',     None             , 960,  1, 1, "same", "HS", None, "Conv2", None],
                         ['Conv',     None             , 1280, 1, 1, "same", "HS", None, "Conv3", None],
                        ]
    unfrozen = ["Conv2", "Conv3","CSP6"]
    
    return Specification, unfrozen

def EfficientNetB0(Dataset=None):
    if Dataset[:5] == "cifar":
                        # 1
        Specification = [['Conv',    None, 32,  3, 1, "same", "SW", None, "Conv1"],
                        # 2
                         ['MBblock', 32  , 16,  3, 1, "same", "SW", None, "MBConv2_1"],
                        # 3
                         ['MBblock', 16  , 24,  3, 1, "same", "SW", None, "MBConv3_1"],
                         ['MBblock', 144 , 24,  3, 1, "same", "SW", None, "MBConv3_2"],
                        # 4
                         ['MBblock', 144 , 40,  5, 2, "same", "SW", None, "MBConv4_1"],
                         ['MBblock', 240 , 40,  5, 1, "same", "SW", None, "MBConv4_2"],
                        # 5
                         ['MBblock', 240 , 80,  3, 2, "same", "SW", None, "MBConv5_1"],
                         ['MBblock', 480 , 80,  3, 1, "same", "SW", None, "MBConv5_2"],
                         ['MBblock', 480 , 80,  3, 1, "same", "SW", None, "MBConv5_3"],
                        # 6
                         ['MBblock', 480 , 112, 5, 2, "same", "SW", None, "MBConv6_1"],
                         ['MBblock', 80  , 112, 5, 1, "same", "SW", None, "MBConv6_2"],
                         ['MBblock', 80  , 112, 5, 1, "same", "SW", None, "MBConv6_3"],
                        # 7
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_1"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_2"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_3"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_4"],
                        # 8
                         ['MBblock', 192 , 320, 3, 1, "same", "SW", None, "MBConv8_1"],
                        ]  
        
    else:
        Specification = [['Conv',    None, 16,  3, 2, "same", "SW", None, "Conv1"],
                        # 2
                         ['MBblock', 32  , 16,  3, 1, "same", "SW", None, "MBConv2_1"],
                        # 3
                         ['MBblock', 16  , 24,  3, 2, "same", "SW", None, "MBConv3_1"],
                         ['MBblock', 16  , 24,  3, 1, "same", "SW", None, "MBConv3_2"],
                        # 4
                         ['MBblock', 24  , 40,  5, 2, "same", "SW", None, "MBConv4_1"],
                         ['MBblock', 24  , 40,  5, 1, "same", "SW", None, "MBConv4_2"],
                        # 5
                         ['MBblock', 40  , 80,  3, 2, "same", "SW", None, "MBConv5_1"],
                         ['MBblock', 40  , 80,  3, 1, "same", "SW", None, "MBConv5_2"],
                         ['MBblock', 40  , 80,  3, 1, "same", "SW", None, "MBConv5_3"],
                        # 6
                         ['MBblock', 80  , 112, 5, 2, "same", "SW", None, "MBConv6_1"],
                         ['MBblock', 80  , 112, 5, 1, "same", "SW", None, "MBConv6_2"],
                         ['MBblock', 80  , 112, 5, 1, "same", "SW", None, "MBConv6_3"],
                        # 7
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_1"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_2"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_3"],
                         ['MBblock', 112 , 192, 5, 1, "same", "SW", None, "MBConv7_4"],
                        # 8
                         ['MBblock', 192 , 320, 3, 1, "same", "SW", None, "MBConv8_1"],
                        ]  
    unfrozen = ["MBConv7_3", "MBConv7_4", "MBConv8_1"]
    return Specification, unfrozen   

def Unet_Encode(Dataset=None):
    Specification = [
        # 512 x 512 x 3
        ['Conv',  None, 32, 3, 1, 'same', 'HS', None, 'Conv1', None],
        ['MBblock', 64, 32, 3, 1, 'same', 'RE', None, 'Mb1'  , None],
        # 256 x 256 x 64
        ['DWconv',None,None,2, 2, 'same','relu',None, 'Pool1', None],
        ['MBblock',128, 64, 3, 1, 'same', 'RE', None, 'Mb2'  , None],
        ['MBblock',128, 64, 3, 1, 'same', 'RE', 'SE', 'Mb3'  , None],
        # 128 x 128 x 128
        ['DWconv',None,None,2, 2, 'same','relu',None, 'Pool2', None],
        ['MBblock',256, 128,3, 1, 'same', 'RE', None, 'Mb4'  , None],
        ['MBblock',256, 128,3, 1, 'same', 'RE', 'SE', 'Mb5'  , None],
        # 64 * 64 * 256
        ['DWconv',None,None,2, 2, 'same','relu',None, 'Pool3', None],
        ['MBblock',512, 256,3, 1, 'same', 'RE', None, 'Mb6'  , None],
        ['MBblock',512, 256,3, 1, 'same', 'RE', 'SE', 'Mb7'  , None],
        # 32 * 32 * 512
        ['DWconv',None,None,2, 2, 'same','relu',None, 'Pool4', None],
        ['MBblock',1024,512,3, 1, 'same', 'RE', None, 'Mb8'  , None],
        ['MBblock',1024,512,3, 1, 'same', 'RE', 'SE', 'Mb9'  , None],
        ]
    unfrozen = []
    return Specification, unfrozen  