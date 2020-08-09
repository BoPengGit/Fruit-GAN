from abc import ABC, abstractmethod
import tensorflow as tf 


class GANLoss(ABC):

    @abstractmethod 
    @classmethod 
    def generator_loss(cls):
        raise NotImplementedError

    @abstractmethod 
    @classmethod 
    def discriminator_loss(cls):
        raise NotImplementedError


class GANCrossEntropyLoss(GANLoss):
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

    @classmethod
    def generator_loss(cls, fake_output, batch_size):
        loss = cls.cross_entropy(tf.ones_like(fake_output), fake_output)
        loss = tf.reduce_sum(loss * (1. / batch_size))
        return loss

    @classmethod
    def discriminator_loss(cls, real_output, fake_output, batch_size):
        real_loss = cls.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cls.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        total_loss = tf.reduce_sum(total_loss * (1. / batch_size))
        return total_loss

