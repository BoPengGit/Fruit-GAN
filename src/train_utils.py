def generate_and_save_images(model, epoch, test_input):
  '''Image generation plot function'''

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(22,22))
  for i in range(predictions.shape[0]):
      plt.subplot(8, 4, i+1)
      plt.imshow(((np.array(predictions[i]) * 127.5) + 127.5).astype(np.uint8))
      plt.axis('off')

  plt.show()

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))