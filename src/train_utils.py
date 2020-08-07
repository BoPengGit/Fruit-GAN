import tensorflow as tf 




def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model([test_input,labs], training=False)

  fig = plt.figure(figsize=(22,22))

  for i in range(predictions.shape[0]):
      plt.subplot(8, 4, i+1)
      plt.imshow(predictions[i, :, :, 1] * 127.5 + 127.5)
      plt.axis('off')

  plt.show()

