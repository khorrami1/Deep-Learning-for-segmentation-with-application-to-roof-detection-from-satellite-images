# Deep Learning for segmentation with application to roof detection from satellite images

In this project, a few images, around 25 images from sattelites and the corresponding mask (label) images are given. We aim to obtain a neural network to detect roofs from satellite images. To do so, we perform the following steps: (Pretraining: see the notebook pre_training.ipynb, training and test: see main_training.ipynb)

# 1- Extending datastore

Making new images from the original images by the technique of augmentation. For more details, see the following link:
Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
Licensed under the terms of the MIT Licence.

We randomly rotate images and crop them using Augmentor module:

p = Augmentor.DataPipeline(images)

p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)

p.rotate_random_90(probability=1)

num_samples = int(1000)

Now we can sample from the pipeline:

augmented_data = p.sample(num_samples)

# 2- Creating a custom unet 

From this module: (from keras_unet.models import custom_unet - https://github.com/karolzak/keras-unet) with the folloing options:

model = custom_unet(
    input_shape=(img_size_x, img_size_y, num_channels_inp),
    use_batch_norm=False,
    num_classes=1,
    filters=8,
    dropout=0.1,
    activation='tanh',
    output_activation='tanh')

Note that the activation function "tanh" is so helpful for this segmentation.  

# 3- Training the model

In 100 epochs with the following options:

num_epoch = 100

batch_size = 32

learning_rate = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss=['MSE'], metrics=['MSE'])

history = model.fit(input_train, output_train, batch_size=batch_size, epochs=num_epoch,
 validation_data=(input_val, output_val))
 
 ![alt text](https://github.com/khorrami1/Deep-Learning-for-segmentation-with-application-to-roof-detection-from-satellite-images/blob/main/loss_epoch.png)
 
# 4- Testing the nmodel: 

model.evaluate(input_val, output_val)

# 5- Saving the model:
directory_model = 'saved_model'

model.save(directory_model, save_format='tf')

# 6- Results:

![alt text](https://github.com/khorrami1/Deep-Learning-for-segmentation-with-application-to-roof-detection-from-satellite-images/blob/main/result.png)

# 7- visualize the model

How to plot keras models using plot_model on Windows10:

We use the plot_model library:
from tensorflow.keras.utils import plot_model

Plot_model requires Pydot and graphviz libraries.

To install Graphviz: 
Download and install the latest version exe
https://gitlab.com/graphviz/graphviz/... 

To check the installation,
go to the command prompt and enter: dot -V

Open Anaconda prompt for the ​desired environment 

pip install pydot
pip install graphviz

tf.keras.utils.plot_model(
    loaded_model, to_file='model.png', show_shapes=True, show_dtype=False,
    show_layer_names=False, rankdir='TB', expand_nested=False, dpi=200,
    layer_range=None, show_layer_activations=False
)

 ![alt text](https://github.com/khorrami1/Deep-Learning-for-segmentation-with-application-to-roof-detection-from-satellite-images/blob/main/model.png)

