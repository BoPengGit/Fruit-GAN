import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


class Generator():
    
    def __init__(self):
        self.maps = None
        self.noise_dim = None 
        self.init = None

        self.model = self.create(self.noise_dim, self.maps, self.init) 

    def create(self, noise_dim, maps, init):
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

    def __repr__(self):
        repr(self.model)

    def __str__(self):
        repr(self.model)


class Discriminator():
    
    def __init__(self):
        self.model = self.create() 

    def create(self, maps):
        image = tf.keras.Input(shape=((64,64,3)))
        label = tf.keras.Input(shape=((1,)))
        x = layers.Embedding(120, 64*64, input_length=1)(label)
        x = layers.Reshape((64,64,1))(x)
        x = layers.concatenate([image,x])
        
        x = layers.Conv2D(MAPS, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(MAPS*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(MAPS*4, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(MAPS*8, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(121, activation='sigmoid')(x)
        x2 = layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=[image,label], outputs=[x,x2])
        return model

    def __repr__(self):
        repr(self.model)

    def __str__(self):
        repr(self.model)

