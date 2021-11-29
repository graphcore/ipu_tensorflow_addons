Optimizers
~~~~~~~~~~

The optimizers contained in the IPU TensorFlow add-ons are drop in
replacements to TensorFlow optimizers. They are functionally the same but have
a number of additional features, which can be used via the optimizer's kwargs.

The precision of any optimizer states within the optimizer can be set
independently of each other and the model parameters. This is particularly
useful when training in mixed precision.

The optimizer update can be outlined, making the optimizer update block
code reusable, which can reduce memory at the expense of passing variables
around.
