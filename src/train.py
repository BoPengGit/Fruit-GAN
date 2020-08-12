import os 
import tensorflow as tf  
from dataset import *
from augmentation import DiffAugment

class Train():

  def __init__(self):
    self.config = None 
    self.model = None 
    self.generator = None 
    self.discriminator = None 

  def __call__(self, config):
    self.config = config
    self.model = self.model()
    self.generator = self.model.create_generator()
    self.discriminator = self.model.create_discriminator()

    dataset = extract_transform_dataset(self.config)
    self.train(dataset, self.config.epochs)

  def train(self, dataset, epochs):
    train_iterator = iter(dataset)
    for epoch in range(epochs):
      self.train_steps(train_iterator, tf.convert_to_tensor(self.config.steps_per_epoch))

  @tf.function
  def train_steps(self, iterator, steps):
    for _ in tf.range(steps):
      strategy.run(self.train_step_function, args=(next(iterator),))

  def train_step_function(self, images, augmentation=None):
    noise = tf.random.normal([self.config.per_replica_batch_size, self.config.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(noise, training=True)
      
      if augmentation:
          real_output = self.discriminator(DiffAugment(images, policy=self.config.augmentation), training=True)
          fake_output = self.discriminator(DiffAugment(generated_images, policy=augmentation), training=True)
      else:
          real_output = self.discriminator(images, training=True)
          fake_output = self.discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)

      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))



class Strategy:

  def __init__(self, strategy):
    pass

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


