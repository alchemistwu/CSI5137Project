import tensorflow as tf

def model_factory(name, pretrain=None, target_size=256):
    """
    getting the keras application model
    :param name: specifying the model
    :param pretrain: if True, load imagenet pretrained weights
    :param target_size:
    :return:
    """
    weights = None
    if pretrain:
        weights = 'imagenet'
    model_dict = {
        'res': tf.keras.applications.ResNet50,
        'vgg': tf.keras.applications.vgg16.VGG16,
        'googLeNet': tf.keras.applications.inception_v3.InceptionV3,
        'dense': tf.keras.applications.DenseNet121,
        'mobile': tf.keras.applications.MobileNetV2
    }
    return model_dict[name](weights=weights, include_top=False, input_shape=(target_size, target_size, 3))

def get_model(name='res', pretrain=True, verbose=False, target_size=256, n_class=9):
    """
    Due to the API limitation, shape of MNIST has to be resized as (32, 32, 3)
    :param pretrain: if Ture, return attack model with ImageNet weights
    :param task: 'mnist' or 'cifar'
    :param verbose: Print model information if True
    :return: keras model
    """
    input = tf.keras.layers.Input(shape=(target_size, target_size, 3))
    m = model_factory(name=name, pretrain=pretrain, target_size=target_size)(input)
    flatten = tf.keras.layers.GlobalMaxPool2D()(m)
    output = tf.keras.layers.Dense(n_class, activation='softmax')(flatten)
    model = tf.keras.Model(inputs=[input], outputs=[output])
    model.compile(loss=['sparse_categorical_crossentropy'], metrics=['acc'])
    if verbose:
        model.summary()
    return model

if __name__ == '__main__':
    get_model(verbose=True)