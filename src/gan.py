import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


class BaseGAN:

    def __init__(self):
        self.name = None 
        self.generator = None 
        self.discriminator = None 

    def __str__(self):
        return repr(f'GAN Name: {self.name}  Generator: {bool(self.generator)} Discriminator: {bool(self.discriminator)}')

    def __repr__(self):
        return repr(f'GAN Name: {self.name}  Generator: {bool(self.generator)} Discriminator: {bool(self.discriminator)}')


class DCGAN(BaseGAN):
    
    def __init__(self):
        self.name = 'DCGAN'
        self.generator = None 
        self.discriminator = None 

    def create_generator(self, noise_dim, maps, init):
        self.generator = DCGANGenerator.create(noise_dim, maps, init)
        return self.generator

    def create_discriminator(self, image_size, maps, init):
        self.discriminator =  DCGANDiscriminator.create(image_size, maps, init)
        return self.discriminator


class DCGANGenerator:

    @staticmethod
    def create(noise_dim, maps, init):
        seed = tf.keras.Input(shape=((noise_dim,)))
        label = tf.keras.Input(shape=((1,)))
        x = layers.Embedding(120, 120, input_length=1,name='emb')(label)
        x = layers.Flatten()(x)
        x = layers.concatenate([seed,x])
        x = layers.Dense(4*4*maps*8, use_bias=False)(x)
        x = layers.Reshape((4, 4, maps*8))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(maps*4, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(maps*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(maps, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False, activation='tanh')(x)

        model = tf.keras.Model(inputs=[seed,label], outputs=x)    
        return model 


class DCGANDiscriminator:

    @staticmethod
    def create(image_size, maps, init):
        image = tf.keras.Input(shape=((image_size,image_size,3)))
        x = layers.Reshape((image_size,image_size,1))(image)
        x = layers.concatenate([image,x])
        
        x = layers.Conv2D(maps, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(maps*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(maps*4, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(maps*8, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=image, outputs=x)
        return model 