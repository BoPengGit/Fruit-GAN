import time
import tensorflow as tf 

class Train():

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

                # Produce images for the GIF as we go
                display.clear_output(wait=True)
                generate_and_save_images(generator,
                                        epoch + 1,
                                        seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, seed)

    @tf.function
    def train_step(self, image_batch):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator([noise, labs], training=True)

            real_output = discriminator([image_batch, labs], training=True)
            fake_output = discriminator([generated_images, labs], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



