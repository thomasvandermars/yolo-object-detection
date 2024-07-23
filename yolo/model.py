from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Input, Add, MaxPooling2D, Input, Reshape

class conv_batchnorm_lkyrelu(Layer):
    def __init__(self, filters, kernels, strides = (1,1), **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters = filters, 
                           kernel_size = kernels, 
                           strides = strides, 
                           padding = 'same', 
                           use_bias = False,
                           name = self.name + "_conv")
        self.bn = BatchNormalization(name = self.name + "_bn")
        self.act = LeakyReLU(alpha = 0.1, name = self.name + "_lky_relu")
        pass
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

class residual_block(Model):
    def __init__(self, filters, **kwargs):
        super(residual_block, self).__init__(**kwargs)
        
        self.lyrs = []
        self.add = Add()
        self.act = LeakyReLU(alpha = 0.3)
        
        for f in filters:
            self.lyrs.append(conv_batchnorm_leakyrelu(filters = f, kernels = 3))
            pass
        
        self.conv1x1 = conv_batchnorm_leakyrelu(filters = filters[-1], kernels = 1)
        
        pass
    
    def call(self, inputs):
        x = inputs
        for lyr in self.lyrs:
            x = lyr(x)
            pass
        y = self.conv1x1(inputs)
        x = self.add([x, y])
        x = self.act(x)        
        return x

class YOLO(Model):
    def __init__(self, params, layers, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        self.lyrs = []
        for layer in layers:
            self.lyrs.append(layer)
            pass
        
        # add a final convolution block to bring down the number of filters to the number we need for reshaping
        self.lyrs.append(Conv2D(filters = params['ANCHORS'].shape[0] * (len(params['CLASSES']) + 5), kernel_size = 1, name = "final_block"))
        
        # add reshape layer
        self.lyrs.append(Reshape(target_shape = (params['GRID_H'], params['GRID_W'], params['ANCHORS'].shape[0], len(params['CLASSES'])+5),
                                 name = "Reshape"))
        
        self.call(Input(shape = (params['IMG_H'], params['IMG_W'], params['CHANNELS'])))
        pass
        
    def call(self, inputs):
        x = inputs
        for layer in self.lyrs:
            x = layer(x)
            pass
        return x