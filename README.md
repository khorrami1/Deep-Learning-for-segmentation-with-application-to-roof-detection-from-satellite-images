# Deep-Learning-for-segmentation-with-application-to-roof-detection-from-satellite-images
Deep Learning for segmentation, with application to roof detection from satellite images

In this project, a few images, around 25 images from sattelites and the corresponding mask (label) images are given. We aim to obtain a neural network to detect roofs from satellite images. To do so, we perform the following steps: (Pretraining: see the notebook pre_training.ipynb, training and test: see main_training.ipynb)

1- Extending datastore, making new images from the original images by the technique of augmentation. For more details, see the following link:
# Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
# Licensed under the terms of the MIT Licence.

We randomly rotate images and crop them using Augmentor module:
p = Augmentor.DataPipeline(images)
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.rotate_random_90(probability=1)
num_samples = int(1000)
# Now we can sample from the pipeline:
augmented_data = p.sample(num_samples)

2- Creating a custom unet from this module: (from keras_unet.models import custom_unet) with the folloing options:
model = custom_unet(
    input_shape=(img_size_x, img_size_y, num_channels_inp),
    use_batch_norm=False,
    num_classes=1,
    filters=8,
    dropout=0.1,
    activation='tanh',
    output_activation='tanh')

Note that the activation function "tanh" is so helpful for this segmentation.  

3- Training the model in 100 epochs with the following options:

num_epoch = 100
batch_size = 32
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=['MSE'], metrics=['MSE'])
history = model.fit(input_train, output_train, batch_size=batch_size, epochs=num_epoch,
 validation_data=(input_val, output_val))
 
4- Testing the nmodel: 
model.evaluate(input_val, output_val)

5- Saving the model:
directory_model = 'saved_model'
model.save(directory_model, save_format='tf')

