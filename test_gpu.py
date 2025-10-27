import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('CUDA built with TF:', tf.test.is_built_with_cuda())
print('GPU Available:', tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU detected:', gpus[0])
    try:
        # Memory growth setup
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('GPU memory growth enabled')
    except:
        print('Failed to set GPU memory growth')
else:
    print('No GPU detected')
