import keras
import sys
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense,Input
from keras.models import Model, Sequential
from keras.engine import InputSpec, Layer
from keras import regularizers
from keras.utils.conv_utils import conv_output_length
from keras import activations
import numpy as np

class Normalized_Correlation_Layer(Layer):
    '''
    This layer does Normalized Correlation.
    
    It needs to take two inputs(layers),
    currently, it only supports the border_mode = 'valid',
    if you need to output the same shape as input, 
    do padding before giving the layer.
    
    '''
    def __init__(self, patch_size=(5,5),
                 dim_ordering='tf',
                 border_mode='same',
                 stride=(1, 1),
                 activation=None,
                 **kwargs):

        if border_mode != 'same':
            raise ValueError('Invalid border mode for Correlation Layer '
                             '(only "same" is supported as of now):', border_mode)
        self.kernel_size = patch_size
        self.subsample = stride
        self.dim_ordering = dim_ordering
        self.border_mode = border_mode
        self.activation = activations.get(activation)
        super(Normalized_Correlation_Layer, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            inp_rows = input_shape[0][1]
            inp_cols = input_shape[0][2]
        else:
            raise ValueError('Only support tensorflow.')
        
        if self.border_mode != "same":
            rows = conv_output_length(inp_rows, self.kernel_size[0],
                                       self.border_mode, 1)
            cols = conv_output_length(inp_cols, self.kernel_size[1],
                                       self.border_mode, 1)
        else:
            rows = inp_rows
            cols = inp_cols
        
        return (input_shape[0][0], rows, cols,self.kernel_size[0]*cols*input_shape[0][-1])
    

    def call(self, x, mask=None):
        input_1, input_2 = x
        stride_row, stride_col = self.subsample
        inp_shape = input_1._keras_shape
        output_shape = self.compute_output_shape([inp_shape, inp_shape])
        
        #hard coded for 5 patch size
        #print(input_1.shape)
        input_1 = K.spatial_2d_padding(input_1, padding = ((2,2),(2,2)))
        input_2 = K.spatial_2d_padding(input_2, padding = ((4,4),(2,2)))
        
        #print(input_1.shape)
        output_row = output_shape[1]
        output_col = output_shape[2]

        """
        orginal shit
        output_row = inp_shape[1] - self.kernel_size[0] + 1
        output_col = inp_shape[2] - self.kernel_size[1] + 1
        """
        output = []
        for k in range(inp_shape[-1]):
            xc_1 = []
            xc_2 = []
            for i in range(2):
                for j in range(output_col):
                    xc_2.append(K.reshape(input_2[:, i:i+5, j:j+5, k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
            for i in range(output_row):
                slice_row = slice(i, i + self.kernel_size[0])
                slice_row2 = slice(i+2, i + self.kernel_size[0]+2)
                for j in range(output_col):
                    slice_col = slice(j, j + self.kernel_size[1])
                    xc_2.append(K.reshape(input_2[:, slice_row2, slice_col, k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
                    if i % stride_row == 0 and j % stride_col == 0:
                        xc_1.append(K.reshape(input_1[:, slice_row, slice_col, k],
                                              (-1, 1,self.kernel_size[0]*self.kernel_size[1])))
            for i in range(output_row, output_row+2):
                for j in range(output_col):
                    xc_2.append(K.reshape(input_2[:, i:i+5, j:j+5, k],
                                          (-1, 1,self.kernel_size[0]*self.kernel_size[1])))

            xc_1_aggregate = K.concatenate(xc_1, axis=1) # batch_size x w'h' x (k**2*d), w': w/subsample-1
            xc_1_mean = K.mean(xc_1_aggregate, axis=-1, keepdims=True)
            xc_1_std = K.std(xc_1_aggregate, axis=-1, keepdims=True)
            xc_1_aggregate = (xc_1_aggregate - xc_1_mean) / xc_1_std
    
            xc_2_aggregate = K.concatenate(xc_2, axis=1) # batch_size x wh x (k**2*d), w: output_row
            xc_2_mean = K.mean(xc_2_aggregate, axis=-1, keepdims=True)
            xc_2_std = K.std(xc_2_aggregate, axis=-1, keepdims=True)
            xc_2_aggregate = (xc_2_aggregate - xc_2_mean) / xc_2_std
            xc_1_aggregate = K.permute_dimensions(xc_1_aggregate, (0, 2, 1))
            block = []
            l_xc_1= len(xc_1)
            l_xc_2= len(xc_2)
            for i in range(l_xc_1):
                sl1 = slice(int(i/12)*12, int(i/12)*12+60)
                block.append(K.reshape(K.batch_dot(xc_2_aggregate[:,sl1,:],
                                      xc_1_aggregate[:,:,i]), (-1,1,1,60)))
            block = K.concatenate(block, axis=1)
            block = K.reshape(block, (-1,37,12,60))
            output.append(block)
        output = K.concatenate(output, axis=-1)
        output = self.activation(output)
        print(output.shape)
        return output

    def get_config(self):
        config = {'patch_size': self.kernel_size,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'stride': self.subsample,
                  'dim_ordering': self.dim_ordering}
        base_config = super(Correlation_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    normalized_layer = Normalized_Correlation_Layer(stride = (1,1), patch_size = (5, 5))([model1, model2])
    x_corr_mod = Model(inputs=[a,b], outputs = normalized_layer)
    try:
        x_corr_mod.summary()
    except:
        pass
    print(x_corr_mod.output._keras_shape)
    return x_corr_mod

def norm_model(input_size = (8,8,2)):
    a = Input(input_size) 
    b = Input(input_size)
    output = Normalized_Correlation_Layer(stride = (1,1), patch_size = (5,5))([a,b])
    return Model(inputs=[a,b], outputs= output)

if __name__ == "__main__":
    import sys
    test_mod = norm_model()
    #test_mod = normalized_X_corr_model()
    try:
        import cv2
        im1  = cv2.imread(sys.argv[1])
        
        #resize as per your needs or create your numpy arrays. image is of 60,160,3 in this case
        X1 = cv2.resize(im1, (60,160))
        im2  = cv2.imread(sys.argv[2])
        X2 = cv2.resize(im2, (60,160))
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X1 = X1.astype('float32')
        X2 = X2.astype('float32')
        
        #this is necessary, dont skip it
        X1 = np.expand_dims(X1, axis= 0)
        X2 = np.expand_dims(X2, axis= 0)
        
        #right now code shits here, first fix it then check the output
        Y1 = test_mod.predict([X1, X2])
        print(Y1.shape)
        #add statement like this
        #np.save(Y1) # check syntax
    except:
        pass
