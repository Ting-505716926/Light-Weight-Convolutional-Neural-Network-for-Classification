# Light-Weight-Convolutional-Neural-Network-for-Classification
本輕量化網路主要基於Cross Stage Partial Network(CSPNet)與MobileNet-V3建構而成，並使用Tensorflow2.x編寫。

This lightweight network is mainly based on Cross Stage Partial Network (CSPNet) and MobileNet-V3, and is written using Tensorflow2.x.

# Prerequisite
- Python 3.8.5
- tensorflow 2.8
- tensorflow-addons 0.16.1

# Dataset
 | Dataset | Link |
 | :---: | :-----: |
 | Cifar10  | https://www.cs.toronto.edu/~kriz/cifar.html |
 | Cifar100 | https://www.cs.toronto.edu/~kriz/cifar.html |
 |Brid 325 <br> (目前已增加到400類) | https://www.kaggle.com/datasets/gpiosenka/100-bird-species |
 | Brid 100 <br> ("Brid 325"前100種類) | https://www.kaggle.com/datasets/gpiosenka/100-bird-species |
  
# Grad-Cam
![image](img/3.png)

# Experimental results

模型所使用的的Attention為SE。

The Attention used by the model is SE.
### Cifar 10
 | Model            | Accuracy | parameters | Phone ms <br>(Qualcomm 865) |
 | :---:            | :------: | :--------: | :-------------------------: |
 | Mobilenetv3Large | 0.8929   |  3,936,296 | 8.3 |
 | Mobilenetv3Small | 0.8727   |  1,556,568 | 6.5 |
 | OurLarge         | 0.8813   |  2,233,069 | 7.8 |
 | OurSmall         | 0.8546   |   923,137  | 5.7 |

### Cifar 100
 | Model            | Accuracy | parameters | Phone ms <br>(Qualcomm 865) |
 | :---:            | :------: | :--------: | :-------------------------: |
 | Mobilenetv3Large | 0.6171   | 4,051,586 | 9.5 |
 | Mobilenetv3Small | 0.5893   | 1,648,818 | 7.1 |
 | OurLarge         | 0.6078   | 2,348,359 | 8.0 |
 | OurSmall         | 0.5737   | 1,015,387 | 6.9 |
 
### Bird-100
 | Model            | Accuracy | parameters |
 | :---:            | :------: | :--------: |
 | Mobilenetv3Large | 0.9620   | 4,051,586 |
 | Mobilenetv3Small | 0.9520   | 1,648,818 |
 | OurLarge         | 0.9380   | 2,348,359 |
 | OurSmall         | 0.9440   | 1,015,387 |
 
### Bird-325
 | Model            | Accuracy | parameters |
 | :---:            | :------: | :--------: |
 | Mobilenetv3Large | 0.9575   | 4,339,811 |
 | Mobilenetv3Small | 0.9489   | 1,879,443 |
 | OurLarge         | 0.9526   | 2,636,728 |
 | OurSmall         | 0.9563   | 1,246,156 |
