# Build-NN-Advanced-nn-API

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. `torchvision.transforms.v2`
      1. `v2.CutMix`
      2. `v2.MixUp`
   2. `torchvision.transforms`
4. nn-model  
   1. **`torch.nn`**
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---

[`torch.nn` API](https://pytorch.org/docs/stable/nn.html)

## lists
- Containers
- Convolution Layers
- Pooling layers
- Padding Layers
- Non-linear Activations (weighted sum, nonlinearity)
- Non-linear Activations (other)
- Normalization Layers
- Recurrent Layers
- Transformer Layers
- Linear Layers
- Dropout Layers
- Sparse Layers
- Distance Functions
- Loss Functions
- Vision Layers
- Shuffle Layers
- DataParallel Layers (multi-GPU, distributed)
- Utilities
- Quantized Functions
- Lazy Modules Initialization

## Containers
- `Module` : 모든 nn module 들의 base class
- `Sequential`
- `ModuleList` : Holds submodules in a list.
- `ModuleDict` : Holds submodules in a dictionary.
- `ParameterList` : Holds parameters in a list.
- `ParameterDict` : Holds parameters in a dictionary.

## Convolution Layers
- `nn.Conv1d` : Applies a 1D convolution over an input signal composed of several input planes.

- `nn.Conv2d` : Applies a 2D convolution over an input signal composed of several input planes.

- `nn.Conv3d` : Applies a 3D convolution over an input signal composed of several input planes.

- `nn.ConvTranspose1d` : Applies a 1D transposed convolution operator over an input image composed of several input planes.

- `nn.ConvTranspose2d` : Applies a 2D transposed convolution operator over an input image composed of several input planes.

- `nn.ConvTranspose3d` : Applies a 3D transposed convolution operator over an input image composed of several input planes.

- `nn.LazyConv1d` : A torch.nn.Conv1d module with lazy initialization of the in_channels argument.

- `nn.LazyConv2d` : A torch.nn.Conv2d module with lazy initialization of the in_channels argument.

- `nn.LazyConv3d` : A torch.nn.Conv3d module with lazy initialization of the in_channels argument.

- `nn.LazyConvTranspose1d` : A torch.nn.ConvTranspose1d module with lazy initialization of the in_channels argument.

- `nn.LazyConvTranspose2d` : A torch.nn.ConvTranspose2d module with lazy initialization of the in_channels argument.

- `nn.LazyConvTranspose3d` : A torch.nn.ConvTranspose3d module with lazy initialization of the in_channels argument.

- `nn.Unfold` : Extracts sliding local blocks from a batched input tensor.

- `nn.Fold` : Combines an array of sliding local blocks into a large containing tensor.

## Pooling Layers
- `nn.MaxPool1d` : Applies a 1D max pooling over an input signal composed of several input planes.

- `nn.MaxPool2d` : Applies a 2D max pooling over an input signal composed of several input planes.

- `nn.MaxPool3d` : Applies a 3D max pooling over an input signal composed of several input planes.

- `nn.MaxUnpool1d` : Computes a partial inverse of MaxPool1d.

- `nn.MaxUnpool2d` : Computes a partial inverse of MaxPool2d.

- `nn.MaxUnpool3d` : Computes a partial inverse of MaxPool3d.

- `nn.AvgPool1d` : Applies a 1D average pooling over an input signal composed of several input planes.

- `nn.AvgPool2d` : Applies a 2D average pooling over an input signal composed of several input planes.

- `nn.AvgPool3d` : Applies a 3D average pooling over an input signal composed of several input planes.

- `nn.FractionalMaxPool2d` : Applies a 2D fractional max pooling over an input signal composed of several input planes.

- `nn.FractionalMaxPool3d` : Applies a 3D fractional max pooling over an input signal composed of several input planes.

- `nn.LPPool1d` : Applies a 1D power-average pooling over an input signal composed of several input planes.

- `nn.LPPool2d` : Applies a 2D power-average pooling over an input signal composed of several input planes.

- `nn.LPPool3d` : Applies a 3D power-average pooling over an input signal composed of several input planes.

- `nn.AdaptiveMaxPool1d` : Applies a 1D adaptive max pooling over an input signal composed of several input planes.

- `nn.AdaptiveMaxPool2d` : Applies a 2D adaptive max pooling over an input signal composed of several input planes.

- `nn.AdaptiveMaxPool3d` : Applies a 3D adaptive max pooling over an input signal composed of several input planes.

- `nn.AdaptiveAvgPool1d` : Applies a 1D adaptive average pooling over an input signal composed of several input planes.

- `nn.AdaptiveAvgPool2d` : Applies a 2D adaptive average pooling over an input signal composed of several input planes.

- `nn.AdaptiveAvgPool3d` : Applies a 3D adaptive average pooling over an input signal composed of several input planes.

## Padding Layers
- `nn.ReflectionPad1d` : Pads the input tensor using the reflection of the input boundary.

- `nn.ReflectionPad2d` : Pads the input tensor using the reflection of the input boundary.

- `nn.ReflectionPad3d` : Pads the input tensor using the reflection of the input boundary.

- `nn.ReplicationPad1d` : Pads the input tensor using replication of the input boundary.

- `nn.ReplicationPad2d` : Pads the input tensor using replication of the input boundary.

- `nn.ReplicationPad3d` : Pads the input tensor using replication of the input boundary.

- `nn.ZeroPad1d` : Pads the input tensor boundaries with zero.

- `nn.ZeroPad2d` : Pads the input tensor boundaries with zero.

- `nn.ZeroPad3d` : Pads the input tensor boundaries with zero.

- `nn.ConstantPad1d` : Pads the input tensor boundaries with a constant value.

- `nn.ConstantPad2d` : Pads the input tensor boundaries with a constant value.

- `nn.ConstantPad3d` : Pads the input tensor boundaries with a constant value.

- `nn.CircularPad1d` : Pads the input tensor using circular padding of the input boundary.

- `nn.CircularPad2d` : Pads the input tensor using circular padding of the input boundary.

- `nn.CircularPad3d` : Pads the input tensor using circular padding of the input boundary.


## Non-linear Activations (weighted sum, nonlinearity)
- `nn.ELU` : Applies the Exponential Linear Unit (ELU) function, element-wise.

- `nn.Hardshrink` : Applies the Hard Shrinkage (Hardshrink) function element-wise.

- `nn.Hardsigmoid` : Applies the Hardsigmoid function element-wise.

- `nn.Hardtanh` : Applies the HardTanh function element-wise.

- `nn.Hardswish` : Applies the Hardswish function, element-wise.

- `nn.LeakyReLU` : Applies the LeakyReLU function element-wise.

- `nn.LogSigmoid` : Applies the Logsigmoid function element-wise.

- `nn.MultiheadAttention` : Allows the model to jointly attend to information from different representation subspaces.

- `nn.PReLU` : Applies the element-wise PReLU function.

- `nn.ReLU` : Applies the rectified linear unit function element-wise.

- `nn.ReLU6` : Applies the ReLU6 function element-wise.

- `nn.RReLU` : Applies the randomized leaky rectified linear unit function, element-wise.

- `nn.SELU` : Applies the SELU function element-wise.

- `nn.CELU` : Applies the CELU function element-wise.

- `nn.GELU` : Applies the Gaussian Error Linear Units function.

- `nn.Sigmoid` : Applies the Sigmoid function element-wise.

- `nn.SiLU` : Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

- `nn.Mish` : Applies the Mish function, element-wise.

- `nn.Softplus` : Applies the Softplus function element-wise.

- `nn.Softshrink` : Applies the soft shrinkage function element-wise.

- `nn.Softsign` : Applies the element-wise Softsign function.

- `nn.Tanh` : Applies the Hyperbolic Tangent (Tanh) function element-wise.

- `nn.Tanhshrink` : Applies the element-wise Tanhshrink function.

- `nn.Threshold` : Thresholds each element of the input Tensor.

- `nn.GLU` : Applies the gated linear unit function.

## Non-linear Activations (other)

- `nn.Softmin` : Applies the Softmin function to an n-dimensional input Tensor.

- `nn.Softmax` : Applies the Softmax function to an n-dimensional input Tensor.

- `nn.Softmax2d` : Applies SoftMax over features to each spatial location.

- `nn.LogSoftmax` : Applies the log(Softmax(𝑥)) function to an n-dimensional input Tensor.

- `nn.AdaptiveLogSoftmaxWithLoss` : Efficient softmax approximation.

## Normalization Layers

-` nn.BatchNorm1d` : Applies Batch Normalization over a 2D or 3D input.

-` nn.BatchNorm2d` : Applies Batch Normalization over a 4D input.

-` nn.BatchNorm3d` : Applies Batch Normalization over a 5D input.

-` nn.LazyBatchNorm1d` : A torch.nn.BatchNorm1d module with lazy initialization.

-` nn.LazyBatchNorm2d` : A torch.nn.BatchNorm2d module with lazy initialization.

-` nn.LazyBatchNorm3d` : A torch.nn.BatchNorm3d module with lazy initialization.

-` nn.GroupNorm` : Applies Group Normalization over a mini-batch of inputs.

-` nn.SyncBatchNorm` : Applies Batch Normalization over a N-Dimensional input.

-` nn.InstanceNorm1d` : Applies Instance Normalization.

-` nn.InstanceNorm2d` : Applies Instance Normalization.

-` nn.InstanceNorm3d` : Applies Instance Normalization.

-` nn.LazyInstanceNorm1d` : A torch.nn.InstanceNorm1d module with lazy initialization of the num_features argument.

-` nn.LazyInstanceNorm2d` : A torch.nn.InstanceNorm2d module with lazy initialization of the num_features argument.

-` nn.LazyInstanceNorm3d` : A torch.nn.InstanceNorm3d module with lazy initialization of the num_features argument.

-` nn.LayerNorm` : Applies Layer Normalization over a mini-batch of inputs.

-` nn.LocalResponseNorm` : Applies local response normalization over an input signal.

## Recurrent Layers
- `nn.RNNBase` : Base class for RNN modules (RNN, LSTM, GRU).

- `nn.RNN` : Apply a multi-layer Elman RNN with 
tanh or ReLU non-linearity to an input sequence.

- `nn.LSTM` : Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.

- `nn.GRU` : Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

- `nn.RNNCell` : An Elman RNN cell with tanh or ReLU non-linearity.

- `nn.LSTMCell` : A long short-term memory (LSTM) cell.

- `nn.GRUCell` : A gated recurrent unit (GRU) cell.

## Transformer Layers
-`nn.Transformer` : A transformer model.

-`nn.TransformerEncoder` : TransformerEncoder is a stack of N encoder layers.

-`nn.TransformerDecoder` : TransformerDecoder is a stack of N decoder layers.

-`nn.TransformerEncoderLayer` : TransformerEncoderLayer is made up of self-attn and feedforward network.

-`nn.TransformerDecoderLayer` : TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

## Linear Layers
- `nn.Identity` : A placeholder identity operator that is argument-insensitive.

- `nn.Linear` : Applies a linear transformation to the incoming data: y=xAT+b.

- `nn.Bilinear` : Applies a bilinear transformation to the incoming data: y=x1TAx2 +b.

- `nn.LazyLinear` : A torch.nn.Linear module where in_features is inferred.

## Dropout Layers

- `nn.Dropout` : During training, randomly zeroes some of the elements of the input tensor with probability p.

- `nn.Dropout1d` : Randomly zero out entire channels.

- `nn.Dropout2d` : Randomly zero out entire channels.

- `nn.Dropout3d` : Randomly zero out entire channels.

- `nn.AlphaDropout` : Applies Alpha Dropout over the input.

- `nn.FeatureAlphaDropout` : Randomly masks out entire channels.

## Sparse Layers
- `nn.Embedding` : A simple lookup table that stores embeddings of a fixed dictionary and size.

- `nn.EmbeddingBag` : Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

## Distance Functions

- `nn.CosineSimilarity` : Returns cosine similarity between x1 and x2 computed along dim.

- `nn.PairwiseDistance` : Computes the pairwise distance between input vectors, or between columns of input matrices.

## Loss Functions

- `nn.L1Loss` : Creates a criterion that measures the mean absolute error (MAE) between each element in the input 
𝑥
x and target 
𝑦
y.

- `nn.MSELoss` : Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input 
𝑥
x and target 
𝑦
y.

- `nn.CrossEntropyLoss` : This criterion computes the cross entropy loss between input logits and target.

- `nn.CTCLoss` : The Connectionist Temporal Classification loss.

- `nn.NLLLoss` : The negative log likelihood loss.

- `nn.PoissonNLLLoss` : Negative log likelihood loss with Poisson distribution of target.

- `nn.GaussianNLLLoss` : Gaussian negative log likelihood loss.

- `nn.KLDivLoss` : The Kullback-Leibler divergence loss.

- `nn.BCELoss` : Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:

- `nn.BCEWithLogitsLoss` : This loss combines a Sigmoid layer and the BCELoss in one single class.

- `nn.MarginRankingLoss` : Creates a criterion that measures the loss given inputs 
𝑥
1
x1, 
𝑥
2
x2, two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor 
𝑦
y (containing 1 or -1).

- `nn.HingeEmbeddingLoss` : Measures the loss given an input tensor 
𝑥
x and a labels tensor 
𝑦
y (containing 1 or -1).

- `nn.MultiLabelMarginLoss` : Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input 
𝑥
x (a 2D mini-batch Tensor) and output 
𝑦
y (which is a 2D Tensor of target class indices).

- `nn.HuberLoss` : Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.

- `nn.SmoothL1Loss` : Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.

- `nn.SoftMarginLoss` : Creates a criterion that optimizes a two-class classification logistic loss between input tensor 
𝑥
x and target tensor 
𝑦
y (containing 1 or -1).

- `nn.MultiLabelSoftMarginLoss` : Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input 
𝑥
x and target 
𝑦
y of size 
(
𝑁
,
𝐶
)
(N,C).

- `nn.CosineEmbeddingLoss` : Creates a criterion that measures the loss given input tensors 
𝑥
1
x 
1
​
 , 
𝑥
2
x 
2
​
  and a Tensor label 
𝑦
y with values 1 or -1.

- `nn.MultiMarginLoss` : Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input 
𝑥
x (a 2D mini-batch Tensor) and output 
𝑦
y (which is a 1D tensor of target class indices, 
0
≤
𝑦
≤
x.size
(
1
)
−
1
0≤y≤x.size(1)−1):

- `nn.TripletMarginLoss` : Creates a criterion that measures the triplet loss given an input tensors x1,x2,x3 and a margin with a value greater than 0.

- `nn.TripletMarginWithDistanceLoss` : Creates a criterion that measures the triplet loss given input tensors 𝑎, 𝑝, and n (representing anchor, positive, and negative examples, respectively), and a nonnegative, real-valued function ("distance function") used to compute the relationship between the anchor and positive example ("positive distance") and the anchor and negative example ("negative distance").

## Vision Layers

- `nn.PixelShuffle` : Rearrange elements in a tensor according to an upscaling factor.

- `nn.PixelUnshuffle` : Reverse the PixelShuffle operation.

- `nn.Upsample` : Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

- `nn.UpsamplingNearest2d` : Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.

- `nn.UpsamplingBilinear2d` : Applies a 2D bilinear upsampling to an input signal composed of several input channels.

## Shuffle Layers
- `nn.ChannelShuffle` : Divides and rearranges the channels in a tensor.

## DataParallel Layers (multi-GPU, distributed)
- `nn.DataParallel` : Implements data parallelism at the module level.

- `nn.parallel.DistributedDataParallel` : Implement distributed data parallelism based on torch.distributed at module level.

## Utilities
Utility functions to clip parameter gradients.

- ` clip_grad_norm_` : Clip the gradient norm of an iterable of parameters.

- ` clip_grad_norm` : Clip the gradient norm of an iterable of parameters.

- ` clip_grad_value_` : Clip the gradients of an iterable of parameters at specified value.

Utility functions to flatten and unflatten Module parameters to and from a single vector.

- ` parameters_to_vector` : Flatten an iterable of parameters into a single vector.

- ` vector_to_parameters` : Copy slices of a vector into an iterable of parameters.

Utility functions to fuse Modules with BatchNorm modules.

- ` fuse_conv_bn_eval` : Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.

- ` fuse_conv_bn_weights` : Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

- ` fuse_linear_bn_eval` : Fuse a linear module and a BatchNorm module into a single, new linear module.

- ` fuse_linear_bn_weights` : Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

Utility functions to convert Module parameter memory formats.

- ` convert_conv2d_weight_memory_format` : Convert memory_format of nn.Conv2d.weight to memory_format.

- ` convert_conv3d_weight_memory_format` : Convert memory_format of nn.Conv3d.weight to memory_format The conversion recursively applies to nested nn.Module, including module.

Utility functions to apply and remove weight normalization from Module parameters.

- ` weight_norm` : Apply weight normalization to a parameter in the given module.

- ` remove_weight_norm` : Remove the weight normalization reparameterization from a module.

- ` spectral_norm` : Apply spectral normalization to a parameter in the given module.

- ` remove_spectral_norm` : Remove the spectral normalization reparameterization from a module.

Utility functions for initializing Module parameters.

- ` skip_init` : Given a module class object and args / kwargs, instantiate the module without initializing parameters / buffers.

Utility classes and functions for pruning Module parameters.

- `prune.BasePruningMethod` : Abstract base class for creation of new pruning techniques.

- `prune.PruningContainer` : Container holding a sequence of pruning methods for iterative pruning.

- `prune.Identity` : Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones.

- `prune.RandomUnstructured` : Prune (currently unpruned) units in a tensor at random.

- `prune.L1Unstructured` : Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

- `prune.RandomStructured` : Prune entire (currently unpruned) channels in a tensor at random.

- `prune.LnStructured` : Prune entire (currently unpruned) channels in a tensor based on their Ln-norm.

- `prune.CustomFromMask` : prune.identity` : Apply pruning reparametrization without pruning any units.

- `prune.random_unstructured` : Prune tensor by removing random (currently unpruned) units.

- `prune.l1_unstructured` : Prune tensor by removing units with the lowest L1-norm.

- `prune.random_structured` : Prune tensor by removing random channels along the specified dimension.

- `prune.ln_structured` : Prune tensor by removing channels with the lowest Ln-norm along the specified dimension.

- `prune.global_unstructured` : Globally prunes tensors corresponding to all parameters in parameters by applying the specified pruning_method.

- `prune.custom_from_mask` : Prune tensor corresponding to parameter called name in module by applying the pre-computed mask in mask.

- `prune.remove` : Remove the pruning reparameterization from a module and the pruning method from the forward hook.

- `prune.is_pruned` : Check if a module is pruned by looking for pruning pre-hooks.

Parametrizations implemented using the new parametrization functionality in torch.nn.utils.parameterize.register_parametrization().

- `parametrizations.orthogonal` : Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.

- `parametrizations.weight_norm` : Apply weight normalization to a parameter in the given module.

- `parametrizations.spectral_norm` : Apply spectral normalization to a parameter in the given module.

Utility functions to parametrize Tensors on existing Modules. Note that these functions can be used to parametrize a given Parameter or Buffer given a specific function that maps from an input space to the parametrized space. They are not parameterizations that would transform an object into a parameter. See the Parametrizations tutorial for more information on how to implement your own parametrizations.

- `parametrize.register_parametrization` : Register a parametrization to a tensor in a module.

- `parametrize.remove_parametrizations` : Remove the parametrizations on a tensor in a module.

- `parametrize.cached` : Context manager that enables the caching system within parametrizations registered with register_parametrization().

- `parametrize.is_parametrized` : Determine if a module has a parametrization.

- `parametrize.ParametrizationList` : A sequential container that holds and manages the original parameters or buffers of a parametrized torch.nn.Module.

Utility functions to call a given Module in a stateless manner.

- `stateless.functional_call` : Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.

Utility functions in other modules

- `nn.utils.rnn.PackedSequence` : Holds the data and list of batch_sizes of a packed sequence.

- `nn.utils.rnn.pack_padded_sequence` : Packs a Tensor containing padded sequences of variable length.

- `nn.utils.rnn.pad_packed_sequence` : Pad a packed batch of variable length sequences.

- `nn.utils.rnn.pad_sequence` : Pad a list of variable length Tensors with padding_value.

- `nn.utils.rnn.pack_sequence` : Packs a list of variable length Tensors.

- `nn.utils.rnn.unpack_sequence` : Unpack PackedSequence into a list of variable length Tensors.

- `nn.utils.rnn.unpad_sequence` : Unpad padded Tensor into a list of variable length Tensors.

- `nn.Flatten` : Flattens a contiguous range of dims into a tensor.

- `nn.Unflatten` : Unflattens a tensor dim expanding it to a desired shape.


## Quantized Functions

## Lazy Modules Initialization