# Neural Net Tips
## A compilation of practical tips and advice for building Neural Networks

### Andrej karpathy - Most Common Neural Net Mistakes [source](https://twitter.com/karpathy/status/1013244313327681536)
1. You didn't try to overfit a single batch first.
2. You forgot to toggle train/eval mode for the net. (PyTorch specific, maybe)
3. You forgot to .zero_grad() before .backward() (PyTorch)
4. You passed softmaxed outputs to a loss that expects raw logits.
5. You didn't use bias=False for your Linear/Conv2d layer when using BatchNorm, or conversly forgot to include it for the output layer. This one won't make you silently fail, but they are spurious parameters.
6. Thinking view() and permute() are the same thing (& incorrectly using view).
