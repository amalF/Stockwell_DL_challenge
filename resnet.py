import tensorflow as tf

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5

class BasicBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, stride=1, weight_decay=0.0):

        super(BasicBlock, self).__init__()

        filters1, filters2 = filters
        self.kernel_size = kernel_size
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        l2_regularizer = None
        if weight_decay>0.0:
            l2_regularizer = tf.keras.regularizers.l2(l=weight_decay)

        self.conv1 = tf.keras.layers.Conv2D(filters1, kernel_size, strides=stride,
                    padding='same', use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_regularizer,
                    name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2a')

        self.conv2 = tf.keras.layers.Conv2D(filters2, kernel_size,
                    padding='same', use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_regularizer,
                    name=conv_name_base + '2b')

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2b')

        self.shortcut = tf.keras.Sequential()

        if stride !=1:
            self.shortcut.add(
                    tf.keras.layers.Conv2D(filters2, (1, 1),
                        strides=stride, use_bias=False,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '1'))

            self.shortcut.add(
                    tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                        epsilon=BATCH_NORM_EPSILON,
                        name=bn_name_base + '1')
                    )

    def call(self, inputs):
        x = tf.keras.layers.ReLU()(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))

        shortcut = self.shortcut(inputs)

        x = tf.keras.layers.add([x,shortcut])
        return tf.keras.layers.ReLU()(x)

class BottleneckBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, stride=1, weight_decay=0.0):

        super(BasicBlock, self).__init__()

        filters1, filters2 = filters
        self.kernel_size = kernel_size
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        l2_regularizer = None
        if weight_decay>0.0:
            l2_regularizer = tf.keras.regularizers.l2(l=weight_decay)

        self.conv1 = tf.keras.layers.Conv2D(filters1, 1,
                    padding='same', use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_regularizer,
                    name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2a')

        self.conv2 = tf.keras.layers.Conv2D(filters1, kernel_size, stride=stride,
                    padding='same', use_bias=False,
                    kernel_regularizer=l2_regularizer,
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2b')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2b')

        self.conv3 = tf.keras.layers.Conv2D(filters2, 1,
                    padding='same', use_bias=False,
                    kernel_regularizer=l2_regularizer,
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2c')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON,
                name=bn_name_base + '2c')



        self.shortcut = tf.keras.Sequential()

        if stride !=1:
            self.shortcut.add(
                    tf.keras.layers.Conv2D(filters2, (1, 1),
                        strides=stride, use_bias=False,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '1'))

            self.shortcut.add(
                    tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                        epsilon=BATCH_NORM_EPSILON,
                        name=bn_name_base + '1')
                    )

    def call(self, inputs):
        x = tf.keras.layers.ReLU()(self.bn1(self.conv1(inputs)))
        x = tf.keras.layers.ReLU()(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        shortcut = self.shortcut(inputs)

        x = tf.keras.layers.add([x,shortcut])
        return tf.keras.layers.ReLU()(x)



class ResNet(tf.keras.Model):
    def __init__(self, num_blocks, block, num_classes=10, weight_decay=0.0):
        super(ResNet, self).__init__()

        l2_regularizer = None
        if weight_decay>0.0:
            l2_regularizer = tf.keras.regularizers.l2(l=weight_decay)


        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), strides=(1, 1),
                    padding='valid', use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2_regularizer,
                    name='conv1')

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY,
                   epsilon=BATCH_NORM_EPSILON,
                   name='bn_conv1')

        self.block1 = self._resnet_block(block, num_blocks[0], 3, [16,16], stage=2, stride=1,weight_decay=weight_decay)
        self.block2 = self._resnet_block(block, num_blocks[1], 3, [32,32], stage=3, stride=2, weight_decay=weight_decay)
        self.block3 = self._resnet_block(block, num_blocks[2], 3, [64,64], stage=4, stride=2, weight_decay=weight_decay)
        self.block4 = self._resnet_block(block, num_blocks[3], 3, [128,128], stage=5, stride=2, weight_decay=weight_decay)
        self.linear = tf.keras.layers.Dense(num_classes, activation="softmax")


    def _resnet_block(self, block, num_blocks, kernel_size, filters, stage, stride=2, weight_decay=0.0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(kernel_size, filters, stage, "block_{}".format(i),stride=stride, weight_decay=weight_decay))
        return tf.keras.Sequential(layers)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.keras.layers.Lambda(lambda x:tf.reduce_mean(x, [1, 2]), name="global_average_pool")(x)


    def call(self, inputs):
        x = tf.keras.layers.ReLU()(self.bn1(self.conv1(inputs)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self._global_avg_pool(x)
        x = self.linear(x)
        return x

def ResNet18(num_classes=10, weight_decay=0.0):
    return ResNet([2,2,2,2], BasicBlock, num_classes=num_classes, weight_decay=weight_decay)

def ResNet34(num_classes=10, weight_decay=0.0):
    return ResNet([3,4,6,3], BasicBlock, num_classes=num_classes, weight_decay=weight_decay)

def ResNet50(num_classes=10, weight_decay=0.0):
    return ResNet([3,4,6,3],Bottleneck,num_classes=num_classes, weight_decay=weight_decay)
                                        
def ResNet101(num_classes=10, weight_decay=0.0):                        
    return ResNet([3,4,23,3],Bottleneck,num_classes=num_classes, weight_decay=weight_decay)
                                        
def ResNet152(num_classes=10, weight_decay=0.0):                        
    return ResNet([3,8,36,3], Bottleneck,num_classes=num_classes, weight_decay=weight_decay)


