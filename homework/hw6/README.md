

You will submit your code along with a pdf document containing a few things.

Choose 5-6 pictures of generated images to show how training progresses (for example - epoch 0, 100, 200, 300, 400, 500)

From this part. A batch of real images, a batch of the gradients from an alternate class for these images, and the modified images the discriminator incorrectly classifies.

From this part. Synthetic images maximizing the class output. One for the discriminator trained without the generator and one for the discriminator trained with the generator.

From this part. Synthetic images maximizing a particular layer of features. Do this for at least two different layers (for example - layer 4 and layer 8.)
Report your test accuracy for the two discriminators.

## Part 1 - Training a GAN on CIFAR10
> Choose 5-6 pictures of generated images to show how training progresses (for example - epoch 0, 100, 200, 300, 400, 500)

### Epoch 0


<div style="page-break-after: always;"></div>

### Epoch 30




<div style="page-break-after: always;"></div>

## Part 2

### Section 1
```
==> Loading data...
	number of workers: 16
==> Loding discriminator model trained without the generator...
==> Sampling a batch...
==> Section 1: Pertube real images
=> Evaluating real images...
Accuracy for real images: 92.1875
=> Evaluating fake images...
Accuracy for fake images: 15.625
```