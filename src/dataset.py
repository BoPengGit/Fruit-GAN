import tensorflow as tf 

def tfrecord_parallel_dataset_extract(path):
    dataset = tf.data.Dataset.list_files(path)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=tf.data.experimental.AUTOTUNE,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    options = tf.data.Options() 
    options.experimental_deterministic= False 
    dataset = dataset.with_options(options)
    return dataset 

def parallel_dataset_transform(dataset, image_size, label, batch_size, buffer_size, image_feature_description):
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(transform_image_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if label:
        dataset = dataset.filter(lambda x,y: tf.equal(y, label))

    dataset = dataset.map(lambda x, y: x, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset 


def transform_image_record(record_image, image_feature_description, image_size):
  features = tf.io.parse_example(record_image, image_feature_description)
  label = features['label']
  image = features['image']
  image = tf.image.random_crop(image,size=[image_size,image_size,3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = (image  - 127.5) / 127.5
  return image, label

def extract_transform_dataset(path, dataset, image_size, label, batch_size, buffer_size, image_feature_description):
    dataset = tfrecord_parallel_dataset_extract(path)
    dataset = parallel_dataset_transform(dataset, image_size, label, batch_size, buffer_size, image_feature_description)
    return dataset

