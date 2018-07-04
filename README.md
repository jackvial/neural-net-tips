# Neural Net Tips
## A compilation of practical tips and advice for building Neural Networks

## Andrej karpathy - Most Common Neural Net Mistakes [source](https://twitter.com/karpathy/status/1013244313327681536)
1. You didn't try to overfit a single batch first.
2. You forgot to toggle train/eval mode for the net. (PyTorch specific, maybe)
3. You forgot to .zero_grad() before .backward() (PyTorch)
4. You passed softmaxed outputs to a loss that expects raw logits.
5. You didn't use bias=False for your Linear/Conv2d layer when using BatchNorm, or conversly forgot to include it for the output layer. This one won't make you silently fail, but they are spurious parameters.
6. Thinking view() and permute() are the same thing (& incorrectly using view).


## Perception, Control, Cognition (Matt H and Daniel R) - Practical Advice for Building Deep Neural Networks [source](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/amp/?__twitter_impression=true)

In our machine learning lab, we’ve accumulated tens of thousands of training hours across numerous high-powered machines. The computers weren’t the only ones to learn a lot in the process, though: we ourselves have made a lot of mistakes and fixed a lot of bugs.

Here we present some practical tips for training deep neural networks based on our experiences (rooted mainly in TensorFlow). Some of the suggestions may seem obvious to you, but they weren’t to one of us at some point. Other suggestions may not apply or might even be bad advice for your particular task: use discretion!

We acknowledge these are all well-known methods. We, too, stand on the shoulders of giants here! Our objective with this article is simply to summarize them at a high level for use in practice.

### General Tips
- Use the ADAM optimizer. It works really well. Prefer it to more traditional optimizers such as vanilla gradient descent. TensorFlow note: If saving and restoring weights, remember to set up the Saver after setting up the AdamOptimizer, because ADAM has state (namely per-weight learning rates) that need to be restored as well.
- ReLU is the best nonlinearity (activation function). Kind of like how Sublime is the best text editor. But really, ReLUs are fast, simple, and, amazingly, they work, without diminishing gradients along the way. While sigmoid is a common textbook activation function, it does not propagate gradients well through DNNs.
- Do NOT use an activation function at your output layer. This should be obvious, but it is an easy mistake to make if you build each layer with a shared function: be sure to turn off the activation function at the output.
- DO add a bias in every layer. This is ML 101: a bias essentially translates a plane into a best-fitting position. In y=mx+b, b is the bias, allowing the line to move up or down into the “best fit” position.
- Use variance-scaled initialization. In Tensorflow, this looks like tf.contrib.layers.variance_scaling_initializer(). In our experience, this generalizes/scales better than regular Gaussian, truncated normal, and Xavier. Roughly speaking, the variance scaling initializer adjusts the variance the initial random weights based on the number of inputs or outputs at each layer (default in TensorFlow is number of inputs), thus helping signals to propagate deeper into the network without extra “hacks” like clipping or batch normalization. Xavier is similar, except that the variance is nearly the same in all layers; but networks with layers that vary greatly in their shapes (common with convolutional networks) may not cope as well with the same variance in each layer.
- Whiten (normalize) your input data. For training, subtract the mean of the data set, then divide by its standard deviation. The less your weights have to be stretched and pulled in every which direction, the faster and more easily your network will learn. Keeping the input data mean-centered with constant variance will help with this. You’ll have to perform the same normalization to each test input as well, so make sure your training set resembles real data.
- Scale input data in a way that reasonably preserves its dynamic range. This is related to normalization but should happen before normalizing. For example, data x with an actual real-world range of [0, 140000000] can often be tamed with tanh(x) or tanh(x/C) where C is some constant that stretches the curve to fit more of the input range within the dynamic, sloping part of the tanh function. Especially in cases where your input data may be unbounded on one or both ends, the neural net will learn much better between (0,1).
- Don’t bother decaying the learning rate (usually). Learning rate decay was more common with SGD, but ADAM takes care of this naturally. If you absolutely want to squeeze out every ounce of performance: decay the learning rate for a short time at the end of training; you’ll probably see a sudden, very small drop in error, then it will flatten out again.
- If your convolution layer has 64 or 128 filters, that’s probably plenty. Especially for a deep network. Like, really, 128 is A LOT. If you already have a high number of filters, adding more probably won’t improve things.
- Pooling is for transform invariance. Pooling essentially lets the network learn “the general idea” of “that part” of an image. Max pooling, for example, can help a convolutional network become robust against translation, rotation, and scaling of features in the image.

### Debugging a Neural Network
If your network isn’t learning (meaning: the loss/accuracy is not converging during training, or you’re not getting results you expect), try these tips:

- __Overfit!__ The first thing to do if your network isn’t learning is to overfit a training point. Accuracy should be essentially 100% or 99.99%, or an error as close to 0. If your neural network can’t overfit a single data point, something is seriously wrong with the architecture, but it may be subtle. If you can overfit one data point but training on a larger set still does not converge, try the following suggestions.
- Lower your learning rate. Your network will learn slower, but it may find its way into a minimum that it couldn’t get into before because its step size was too big. (Intuitively, think of stepping over a ditch on the side of the road, when you actually want to get into the lowest part of the ditch, where your error is the lowest.)
- Raise your learning rate. This will speed up training which helps tighten the feedback loop, meaning you’ll have an inkling sooner whether your network is working. While the network should converge sooner, its results probably won’t be great, and the “convergence” might actually jump around a lot. (With ADAM, we found ~0.001 to be pretty good in many experiences.)
- Decrease (mini-)batch size. Reducing a batch size to 1 can give you more granular feedback related to the weight updates, which you should report with TensorBoard (or some other debugging/visualization tool).
- Remove batch normalization. Along with decreasing batch size to 1, doing this can expose diminishing or exploding gradients. For weeks we had a network that wasn’t converging, and only when we removed batch normalization did we realize that the outputs were all NaN by the second iteration. Batch norm was putting a band-aid on something that needed a tourniquet. It has its place, but only after you know your network is bug-free.
- Increase (mini-)batch size. A larger batch size—heck, the whole training set if you could—reduces variance in gradient updates, making each iteration more accurate. In other words, weight updates will be in the right direction. But! There’s an effective upper bound on its usefulness, as well as physical memory limits. Typically, we find this less useful than the previous two suggestions to reduce batch size to 1 and remove batch norm.
- Check your reshaping. Drastic reshaping (like changing an image’s X,Y dimensions) can destroy spatial locality, making it harder for a network to learn since it must also learn the reshape. (Natural features become fragmented. The fact that natural features appear spatially local is why conv nets are so effective!) Be especially careful if reshaping with multiple images/channels; use numpy.stack() for proper alignment.
- Scrutinize your loss function. If using a complex function, try simplifying it to something like L1 or L2. We’ve found L1 to be less sensitive to outliers, making less drastic adjustments when hitting a noisy batch or training point.
- Scrutinize your visualizations, if applicable. Is your viz library (matplotlib, OpenCV, etc.) adjusting the scale of the values, or clipping them? Consider using a perceptually-uniform color scheme as well.
