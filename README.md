# FSPool: Featurewise sort pooling

![Sketch of how the pooling method works](overview.png)

This is the official implementation of our paper [FSPool: Learning Set Representations with Featurewise Sort Pooling][0] in PyTorch.
We propose a pooling method for sets of feature vectors that allows deep neural networks to learn better set representations, classification results and convergence speed.

We also analyse why existing set auto-encoders struggle to auto-encode some very simple sets and attribute this to a responsibility problem.
FSPool can be used to construct a permutation-equivariant auto-encoder, which avoids this problem and results in much better reconstructions and representations.

The main algorithm is located in the stand-alone [`fspool.py`][1] file in this top-level directory, which is the only file you need if you want to use FSPool.
The only dependency for this is PyTorch 1.0 or newer.
Please refer to the READMEs in the `auto-encoder` and `clevr` directories for instructions on reproducing the individual experiments in the paper.

## BibTeX entry

```
@Article{Zhang2019FSPool,
  author        = {Yan Zhang and Jonathon Hare and Adam Pr\"ugel-Bennett},
  title         = {{FSPool}: Learning Set Representations with Featurewise Sort Pooling},
  year          = {2019},
  eprint        = {1906.02795},
  url           = {https://arxiv.org/abs/1906.02795}
}
```

[0]: https://arxiv.com/abs/1906.02795
[1]: https://github.com/Cyanogenoid/fspool/blob/master/fspool.py
