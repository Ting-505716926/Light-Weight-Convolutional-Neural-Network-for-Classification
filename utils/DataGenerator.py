from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras import utils
import numpy as np

def DataGenerator():
    datagen_train = ImageDataGenerator(
        rescale=1/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen_val = ImageDataGenerator(
        rescale=1/255,
    )
    return datagen_train, datagen_val


def load_data(Dataset="cifar100", dir="./Dataset/", input_shape=None, BATCH_SIZE=32, num_classes=None):
    datagen_train, datagen_val = DataGenerator()
    
    if Dataset[:5] == "cifar":
        if Dataset == "cifar100":
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=123)

        for train_index, val_index in sss.split(x_train, y_train):
            X_train, X_val = x_train[train_index], x_train[val_index]
            Y_train, Y_val = y_train[train_index], y_train[val_index]
        print(np.shape(y_test))
        Y_train = utils.to_categorical(Y_train, num_classes)
        Y_val = utils.to_categorical(Y_val, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)

        print("Number of training samples: ", Y_train.shape[0])
        print("Number of validation samples: ", Y_val.shape[0])
        return datagen_train, datagen_val, X_train, Y_train, X_val, Y_val, x_test, y_test
    else:
        trainset = datagen_train.flow_from_directory(dir + 'train',
                                                    target_size=(input_shape[0], input_shape[1]),
                                                    batch_size=BATCH_SIZE,
                                                    # save_to_dir='./Dataset/Aug_train'
                                                    )
        valset = datagen_val.flow_from_directory(dir + 'valid',
                                                target_size=(input_shape[0], input_shape[1]),
                                                batch_size=BATCH_SIZE,
                                                # save_to_dir='./Dataset/Aug_val'
                                                )
        testset = datagen_val.flow_from_directory(dir + 'test',
                                                target_size=(input_shape[0], input_shape[1]),
                                                batch_size=BATCH_SIZE,
                                                # save_to_dir='./Dataset/Aug_val'
                                                )
        return trainset, valset, testset
