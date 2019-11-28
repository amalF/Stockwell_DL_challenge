import tensorflow as tf

def get_optimizer(optimizer_name, lr):
    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    elif optimizer_name == "ADAM":
        return tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-2)


