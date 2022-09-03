# Ambigrams_dual_discrim_GAN

An ambigram is a series of characters that displays one message rightside-up and another (or same) message upside-down.<br>

<br>An example (drawn on my iPad):

<img src="https://github.com/jdowner212/Ambigrams_dual_discrim_GAN/blob/main/sample_images/one.png" width="300" height="200" />
<img src="https://github.com/jdowner212/Ambigrams_dual_discrim_GAN/blob/main/sample_images/two.png" width="300" height="200" />


### Drawing ambigrams with neural networks

The example above is simple, but as ambigrams lengthen, they get harder to produce by hand. Can a neural network do the work for us? 

#### Generative Adversarial Networks (GANs)

A traditional GAN (Generative Adversarial Network) consists of two core parts -- a discriminator network and a generator network. The discriminator is trained to distinguish between 'real' and 'fake' images, and the generator is a convolutional neural network that performs a forward pass on randomly-generated values to produce an image.

In each round of training, the discriminator is provided with 'real' images and 'fake' images (produced by the generator), along with their labels. It guesses whether each image it sees is real or fake and self-updates according to its success rate. The generator is penalized if the image it generates is recognized as fake, and updates itself accordingly. 

#### My version: +1 discriminator

My implementation relies on two discriminators and two datasets -- one for each letter. The first discriminator learns to recognize the first letter rightside-up -- the second does the same for the second letter turned upside-down. The generator penalized against both discriminators, so its updates incorporate information about both letters.

A few examples of images generated by this network.

  <center><img src="https://github.com/jdowner212/Ambigrams_DDGAN/blob/main/sample_images/AB/AB_10_epoch_20_images_2.png" ></center>
  <center><img src="https://github.com/jdowner212/Ambigrams_DDGAN/blob/main/sample_images/RS/RS_28_epoch_13_images_10.png" ></center>
  <center><img src="https://github.com/jdowner212/Ambigrams_DDGAN/blob/main/sample_images/ZE/ZE_49_epoch_6_images_10.png" ></center></center>
   
Of course, whether or not the images 'pass' for the right letters is subjective. The 'generated_samples' folder includes examples of other images I thought passed an arbitrary threshold of acceptability. (Of the three pairs of letters I focused on, the 'R'/'S' images were particularly difficult to generate -- I was a bit more generous in my consideration of those images.)


### Included in this repository

1. `fonts`: contains 600 font files
2. `data`: you will populate this folder yourself with the `add_augmented_data()` function (included in notebook). Generates images of each letter in upper/lower case of every font -- multiplies and transforms results to increase size of dataset.
3. `sample_images`: examples of ambigrams generated by this network
4. `models`: models I've trained with some success, plus whichever you train yourself.
5. `observations_and_tips.txt`: what it sounds like.
6. `Ambigrams.ipynb`: notebook that facilitates data collection, training, and visualization of results. Supported by several `.py` files:
- `my_DDGAN.py`: framework for the network + function to perform updates during training
- `general_utils.py`: brief and boring functions to make our lives easier
- `file_utils.py`: retrieve/manipulate data and organize files
- `train_plot_save.py`: train/save models, display results
