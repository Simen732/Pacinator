import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)

if gpus:
    print('\n✓ GPU DETECTED!')
    print('GPU Name:', gpus[0])
    print('\nYour RTX 3060 is ready to use!')
else:
    print('\n✗ No GPU detected')
    print('Running on CPU only')
