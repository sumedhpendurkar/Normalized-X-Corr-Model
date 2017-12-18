import keras
from keras.layers import Conv2D, MaxPooling2D, Dense,Input
from keras.models import Model, Sequential

def normalized_X_corr_model():
    a = Input((160,60,3))
    b = Input((160,60,3))
    model = Sequential()
    model.add(Conv2D(kernel_size = (5,5), filters = 20,input_shape = (160,60,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(kernel_size = (5,5), filters =  25))
    model.add(MaxPooling2D((2,2)))
    model1 = model(b)
    model2 = model(a)
    try:
        model.summary()
    except:
        pass
    print(model1._keras_shape)
if __name__ == "__main__":
    normalized_X_corr_model()
