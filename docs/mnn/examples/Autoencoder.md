# Autoencoder example

An autoencoder is a type of neural network structure that aims to learn efficiently compressed representations of input dataset. It achieves this by having a network architecture that tapers to some smaller latent space representation, before expanding back to the input size. An autoencoder can generally be thought of as consisting of two networks: the encoder and decoder networks respectively. The encoder is responsible for learning the compressed representation of the input data while the decoder is responsible for reconstructing the input from the latent variable representation.

Autoencoders are useful for a number of tasks such as denoising, feature extraction, data compression and anomaly detection. In this example, we will construct an autoencoder to perform denoising on a sliding window of temporal data.


```python
# Uncomment this to automatically install the moku library in Google Colab or Jupyter Notebook
#%pip install -q moku[neuralnetwork]

# import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from moku.nn import LinnModel, save_linn
except ImportError:
    print("Moku library is not installed.")
    print("If you're running in a Notebook like Google Colab, uncomment the first line in this cell and run it again")
    print("Otherwise install the moku library with neural network extensions using `pip install moku[neuralnetwork]`")
    raise

# set the seed for repeatability
np.random.seed(0)
```

    2024-10-31 14:02:51.344693: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-31 14:02:51.345489: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2024-10-31 14:02:51.347056: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2024-10-31 14:02:51.352066: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-10-31 14:02:51.362040: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-10-31 14:02:51.362050: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-10-31 14:02:51.368955: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-31 14:02:51.803594: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


# Data Generation

As an example use case, we will target a denoising application of an input signal. A useful aspect of autoencoders for this application is that we don't need to know the underlying properties of the noise-less input signal to train our autoencoder. 

We will assume our input signal has some additive white noise we wish to remove from it. We will use a series of random walks as our training data set for this autoencoder. A random walk can be thought of as a summation or integral over a time period of white noise. Training an autoencoder using these random walks allows it to encode these signals in a lower dimension latent space, and then extract useful information from this latent space on real input signals after training. 


```python
# time base, width chosen to be 32
T = np.linspace(-1, 1, 32)
```

We define a random walk with a given step size on a given time base. To achieve this, we generate a random floating point number uniformly distributed between -1 and 1 as our starting point, and then repeatedly add another random floating point number drawn from a Gaussian distribution with zero mean, and standard deviation given by the step size. After each step is taken, we also clip the output to be between -1 and 1, as this is the allowed range of inputs for our network. 


```python
# random walk for training data
def random_walk(step_size, input_array):
    running_value = np.random.uniform(-1,1,1)[0]
    output_array = np.random.normal(0, step_size, input_array.shape)
    output_array[0] = running_value
    for idx, _ in enumerate(output_array):
        if idx != 0:
            output_array[idx] = np.clip(output_array[idx] + output_array[idx - 1], -1, 1)

    return output_array

# generate an example plot
rd_nn = random_walk(0.3, T)
plt.plot(T, rd_nn)
plt.xlabel('Time (arb.)')
plt.ylabel('Voltage (V)')
plt.show()
```


    
![png](Autoencoder_files/Autoencoder_7_0.png)
    


This produces a random walk with step size following a Gaussian distribution. To prepare our training data set, we take a large number of these walks, setting the standard deviation of the step size equal to 0.1. 


```python
# define the length of our training data
data_len = 1000

# pre-define the arrays
training_data = np.zeros((data_len, T.size))

# generate a all of the random waveforms for training and store them
for idx in tqdm(range(data_len)):

    # get the random walks
    Y_train = random_walk(0.1, T)

    training_data[idx, :] = Y_train
```

    100%|██████████| 1000/1000 [00:00<00:00, 13609.92it/s]


# Model definition and training

Now that we have created a training set we will create a model that can be transferred to the Moku after training. Using the provided package we will create an instance of the quantised model. This custom model class will take care of all our quantisation and data scaling.


```python
# create the model object representing the quantized neural network in the Moku
quant_mod = LinnModel()
```

Since we have constructed our training data, we can tell the model what to use for training. This will automatically set up the scaling such that the inputs and outputs are mapped to the domain `[-1, 1]` which will facilitate later quantisation.


```python
quant_mod.set_training_data(training_inputs=training_data, training_outputs=training_data)
```

We now need to define our model structure. To define a structure we will pass a list which contains the model definition. The definition is expected to be a list of tuples of either `(layer_size)` or `(layer_size, layer_activation)`. In our case we are going to define a 4 layer autoencoder which constricts to a latent space size of `2` with the intermediate layers each having a `tanh` activation function. The `show_summary` flag will output the model structure constructed in Tensorflow. There will be a number of intermediate layers for clipping the outputs to conform with the Moku FPGA requirements, but these can largely be ignored. 


```python
# model definition for an autoencoder
model_definition = [(16, 'tanh'), (2, 'tanh'), (16, 'tanh'), (T.size, 'linear')]

# build the model
quant_mod.construct_model(model_definition, show_summary=True)
```

    2024-10-31 14:03:03.565017: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
    2024-10-31 14:03:03.565420: W tensorflow/core/common_runtime/gpu/gpu_device.cc:2251] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">528</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">34</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">48</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_2             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">544</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_3             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,154</span> (4.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,154</span> (4.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



We are now ready to train the model! This can be achieved by simply calling the `fit_model()` function and defining a number of training epochs. For this example we will also specify two additional parameters. The first is `validation_split` which corresponds to the percentage of training data that will be removed from the training process and instead used to validate the performance of the model. Using the validation data we will also define an early stopping configuration with the keyword `es_config`. In this configuration we will define 2 parameters: patience which defines how many epochs we go without seeing model improvement, and, restore which reverts the model to its most performant point.


```python
# define the configuration for the early stopping
early_stopping_config = {'patience': 50, 'restore':True}

# set the training data
history = quant_mod.fit_model(epochs=1000, es_config=early_stopping_config, validation_split=0.1)
```

    Value for monitor missing. Using default:val_loss.


    Epoch 1/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 0.3181 - val_loss: 0.2305
    Epoch 2/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.2113 - val_loss: 0.1483
    Epoch 3/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.1442 - val_loss: 0.0947
    Epoch 4/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.0901 - val_loss: 0.0659
    Epoch 5/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0635 - val_loss: 0.0507
    Epoch 6/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step - loss: 0.0504 - val_loss: 0.0427
    Epoch 7/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.0425 - val_loss: 0.0361
    Epoch 8/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0359 - val_loss: 0.0317
    Epoch 9/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0315 - val_loss: 0.0290
    Epoch 10/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 671us/step - loss: 0.0295 - val_loss: 0.0269
    Epoch 11/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0272 - val_loss: 0.0255
    Epoch 12/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 685us/step - loss: 0.0258 - val_loss: 0.0237
    Epoch 13/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 663us/step - loss: 0.0241 - val_loss: 0.0229
    Epoch 14/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 655us/step - loss: 0.0224 - val_loss: 0.0223
    Epoch 15/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 663us/step - loss: 0.0224 - val_loss: 0.0221
    Epoch 16/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 680us/step - loss: 0.0219 - val_loss: 0.0216
    Epoch 17/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0214 - val_loss: 0.0221
    Epoch 18/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0211 - val_loss: 0.0219
    Epoch 19/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.0215 - val_loss: 0.0210
    Epoch 20/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0210 - val_loss: 0.0207
    Epoch 21/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0208 - val_loss: 0.0207
    Epoch 22/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0211 - val_loss: 0.0204
    Epoch 23/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 809us/step - loss: 0.0203 - val_loss: 0.0203
    Epoch 24/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0200 - val_loss: 0.0203
    Epoch 25/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0193 - val_loss: 0.0202
    Epoch 26/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.0199 - val_loss: 0.0201
    Epoch 27/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 841us/step - loss: 0.0202 - val_loss: 0.0200
    Epoch 28/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 0.0201 - val_loss: 0.0200
    Epoch 29/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0193 - val_loss: 0.0198
    Epoch 30/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.0197 - val_loss: 0.0197
    Epoch 31/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 623us/step - loss: 0.0197 - val_loss: 0.0198
    Epoch 32/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0193 - val_loss: 0.0197
    Epoch 33/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0189 - val_loss: 0.0195
    Epoch 34/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 636us/step - loss: 0.0194 - val_loss: 0.0198
    Epoch 35/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 650us/step - loss: 0.0197 - val_loss: 0.0197
    Epoch 36/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 650us/step - loss: 0.0196 - val_loss: 0.0196
    Epoch 37/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 623us/step - loss: 0.0191 - val_loss: 0.0196
    Epoch 38/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0189 - val_loss: 0.0194
    Epoch 39/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 647us/step - loss: 0.0189 - val_loss: 0.0194
    Epoch 40/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0194 - val_loss: 0.0193
    Epoch 41/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 653us/step - loss: 0.0194 - val_loss: 0.0192
    Epoch 42/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0186 - val_loss: 0.0193
    Epoch 43/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 659us/step - loss: 0.0192 - val_loss: 0.0192
    Epoch 44/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0186 - val_loss: 0.0192
    Epoch 45/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 646us/step - loss: 0.0190 - val_loss: 0.0191
    Epoch 46/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0199 - val_loss: 0.0191
    Epoch 47/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0189 - val_loss: 0.0192
    Epoch 48/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0195 - val_loss: 0.0193
    Epoch 49/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0188 - val_loss: 0.0190
    Epoch 50/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.0192 - val_loss: 0.0191
    Epoch 51/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0184 - val_loss: 0.0189
    Epoch 52/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 713us/step - loss: 0.0184 - val_loss: 0.0189
    Epoch 53/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 630us/step - loss: 0.0184 - val_loss: 0.0190
    Epoch 54/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 632us/step - loss: 0.0185 - val_loss: 0.0189
    Epoch 55/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 641us/step - loss: 0.0181 - val_loss: 0.0193
    Epoch 56/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 647us/step - loss: 0.0189 - val_loss: 0.0189
    Epoch 57/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.0187 - val_loss: 0.0190
    Epoch 58/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 645us/step - loss: 0.0184 - val_loss: 0.0194
    Epoch 59/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 675us/step - loss: 0.0189 - val_loss: 0.0190
    Epoch 60/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 648us/step - loss: 0.0184 - val_loss: 0.0191
    Epoch 61/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.0183 - val_loss: 0.0189
    Epoch 62/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 863us/step - loss: 0.0181 - val_loss: 0.0189
    Epoch 63/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 0.0190 - val_loss: 0.0187
    Epoch 64/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0187 - val_loss: 0.0189
    Epoch 65/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0190 - val_loss: 0.0187
    Epoch 66/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.0183 - val_loss: 0.0188
    Epoch 67/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0184 - val_loss: 0.0187
    Epoch 68/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.0182 - val_loss: 0.0188
    Epoch 69/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 702us/step - loss: 0.0184 - val_loss: 0.0187
    Epoch 70/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.0185 - val_loss: 0.0188
    Epoch 71/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0177 - val_loss: 0.0187
    Epoch 72/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0184 - val_loss: 0.0187
    Epoch 73/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 0.0187 - val_loss: 0.0187
    Epoch 74/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0182 - val_loss: 0.0187
    Epoch 75/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0179 - val_loss: 0.0186
    Epoch 76/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.0178 - val_loss: 0.0187
    Epoch 77/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0190 - val_loss: 0.0186
    Epoch 78/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0186 - val_loss: 0.0190
    Epoch 79/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 682us/step - loss: 0.0194 - val_loss: 0.0186
    Epoch 80/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0181 - val_loss: 0.0186
    Epoch 81/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 661us/step - loss: 0.0188 - val_loss: 0.0186
    Epoch 82/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 854us/step - loss: 0.0184 - val_loss: 0.0189
    Epoch 83/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 0.0189 - val_loss: 0.0187
    Epoch 84/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.0185 - val_loss: 0.0186
    Epoch 85/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 705us/step - loss: 0.0185 - val_loss: 0.0186
    Epoch 86/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0183 - val_loss: 0.0185
    Epoch 87/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0187 - val_loss: 0.0186
    Epoch 88/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 644us/step - loss: 0.0178 - val_loss: 0.0186
    Epoch 89/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 639us/step - loss: 0.0185 - val_loss: 0.0186
    Epoch 90/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 660us/step - loss: 0.0183 - val_loss: 0.0187
    Epoch 91/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 658us/step - loss: 0.0188 - val_loss: 0.0185
    Epoch 92/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 642us/step - loss: 0.0183 - val_loss: 0.0185
    Epoch 93/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 659us/step - loss: 0.0182 - val_loss: 0.0186
    Epoch 94/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 642us/step - loss: 0.0184 - val_loss: 0.0186
    Epoch 95/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 649us/step - loss: 0.0186 - val_loss: 0.0185
    Epoch 96/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 667us/step - loss: 0.0182 - val_loss: 0.0186
    Epoch 97/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 653us/step - loss: 0.0181 - val_loss: 0.0185
    Epoch 98/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 646us/step - loss: 0.0181 - val_loss: 0.0186
    Epoch 99/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 659us/step - loss: 0.0184 - val_loss: 0.0185
    Epoch 100/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 101/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 102/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.0175 - val_loss: 0.0184
    Epoch 103/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 680us/step - loss: 0.0178 - val_loss: 0.0185
    Epoch 104/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 675us/step - loss: 0.0180 - val_loss: 0.0186
    Epoch 105/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 867us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 106/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 713us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 107/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.0186 - val_loss: 0.0185
    Epoch 108/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step - loss: 0.0182 - val_loss: 0.0185
    Epoch 109/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0187 - val_loss: 0.0185
    Epoch 110/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0181 - val_loss: 0.0185
    Epoch 111/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.0177 - val_loss: 0.0185
    Epoch 112/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 676us/step - loss: 0.0190 - val_loss: 0.0184
    Epoch 113/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.0185 - val_loss: 0.0184
    Epoch 114/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 688us/step - loss: 0.0186 - val_loss: 0.0188
    Epoch 115/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 664us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 116/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 691us/step - loss: 0.0180 - val_loss: 0.0184
    Epoch 117/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0179 - val_loss: 0.0184
    Epoch 118/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 693us/step - loss: 0.0181 - val_loss: 0.0186
    Epoch 119/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.0185 - val_loss: 0.0185
    Epoch 120/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 121/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.0185 - val_loss: 0.0185
    Epoch 122/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 672us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 123/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 650us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 124/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0179 - val_loss: 0.0185
    Epoch 125/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.0180 - val_loss: 0.0184
    Epoch 126/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 703us/step - loss: 0.0180 - val_loss: 0.0185
    Epoch 127/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0183 - val_loss: 0.0189
    Epoch 128/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0186 - val_loss: 0.0184
    Epoch 129/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0189 - val_loss: 0.0184
    Epoch 130/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step - loss: 0.0178 - val_loss: 0.0185
    Epoch 131/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0180 - val_loss: 0.0184
    Epoch 132/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 0.0182 - val_loss: 0.0186
    Epoch 133/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.0185 - val_loss: 0.0185
    Epoch 134/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0179 - val_loss: 0.0184
    Epoch 135/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0180 - val_loss: 0.0184
    Epoch 136/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 681us/step - loss: 0.0179 - val_loss: 0.0184
    Epoch 137/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0183 - val_loss: 0.0184
    Epoch 138/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 679us/step - loss: 0.0187 - val_loss: 0.0184
    Epoch 139/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 676us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 140/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 667us/step - loss: 0.0180 - val_loss: 0.0184
    Epoch 141/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0185 - val_loss: 0.0185
    Epoch 142/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.0183 - val_loss: 0.0184
    Epoch 143/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0186 - val_loss: 0.0183
    Epoch 144/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 667us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 145/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0187 - val_loss: 0.0184
    Epoch 146/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0188 - val_loss: 0.0184
    Epoch 147/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 0.0186 - val_loss: 0.0184
    Epoch 148/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.0178 - val_loss: 0.0185
    Epoch 149/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 717us/step - loss: 0.0185 - val_loss: 0.0183
    Epoch 150/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 838us/step - loss: 0.0179 - val_loss: 0.0185
    Epoch 151/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 152/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 153/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.0174 - val_loss: 0.0183
    Epoch 154/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.0183 - val_loss: 0.0184
    Epoch 155/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 684us/step - loss: 0.0182 - val_loss: 0.0184
    Epoch 156/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 157/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 843us/step - loss: 0.0184 - val_loss: 0.0183
    Epoch 158/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 675us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 159/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 684us/step - loss: 0.0184 - val_loss: 0.0183
    Epoch 160/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 695us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 161/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0182 - val_loss: 0.0184
    Epoch 162/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 163/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 164/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 165/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.0186 - val_loss: 0.0183
    Epoch 166/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 681us/step - loss: 0.0184 - val_loss: 0.0185
    Epoch 167/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 168/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0179 - val_loss: 0.0183
    Epoch 169/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0176 - val_loss: 0.0183
    Epoch 170/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.0178 - val_loss: 0.0182
    Epoch 171/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 696us/step - loss: 0.0175 - val_loss: 0.0183
    Epoch 172/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.0182 - val_loss: 0.0185
    Epoch 173/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 174/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 665us/step - loss: 0.0176 - val_loss: 0.0183
    Epoch 175/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 829us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 176/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 723us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 177/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 713us/step - loss: 0.0172 - val_loss: 0.0182
    Epoch 178/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 679us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 179/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 691us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 180/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 181/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 653us/step - loss: 0.0187 - val_loss: 0.0183
    Epoch 182/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0188 - val_loss: 0.0183
    Epoch 183/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 935us/step - loss: 0.0179 - val_loss: 0.0185
    Epoch 184/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 185/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.0183 - val_loss: 0.0184
    Epoch 186/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.0178 - val_loss: 0.0185
    Epoch 187/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 671us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 188/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 696us/step - loss: 0.0179 - val_loss: 0.0183
    Epoch 189/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 702us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 190/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 191/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.0178 - val_loss: 0.0182
    Epoch 192/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0189 - val_loss: 0.0183
    Epoch 193/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 194/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 195/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 196/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 197/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 672us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 198/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 199/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 667us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 200/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 649us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 201/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 661us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 202/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 698us/step - loss: 0.0183 - val_loss: 0.0188
    Epoch 203/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0187 - val_loss: 0.0182
    Epoch 204/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.0185 - val_loss: 0.0182
    Epoch 205/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 206/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 207/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0179 - val_loss: 0.0183
    Epoch 208/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 659us/step - loss: 0.0182 - val_loss: 0.0184
    Epoch 209/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 650us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 210/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0183 - val_loss: 0.0185
    Epoch 211/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 814us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 212/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 695us/step - loss: 0.0182 - val_loss: 0.0185
    Epoch 213/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 696us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 214/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 0.0175 - val_loss: 0.0182
    Epoch 215/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 681us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 216/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 680us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 217/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.0182 - val_loss: 0.0185
    Epoch 218/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 664us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 219/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 673us/step - loss: 0.0175 - val_loss: 0.0182
    Epoch 220/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 673us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 221/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0174 - val_loss: 0.0183
    Epoch 222/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 223/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 224/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 225/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 226/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0177 - val_loss: 0.0184
    Epoch 227/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0184 - val_loss: 0.0185
    Epoch 228/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 662us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 229/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 644us/step - loss: 0.0185 - val_loss: 0.0183
    Epoch 230/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 657us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 231/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 674us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 232/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.0175 - val_loss: 0.0183
    Epoch 233/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 982us/step - loss: 0.0187 - val_loss: 0.0181
    Epoch 234/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 796us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 235/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 236/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0185 - val_loss: 0.0183
    Epoch 237/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 238/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 239/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 240/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 241/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 717us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 242/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 836us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 243/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 244/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 804us/step - loss: 0.0178 - val_loss: 0.0182
    Epoch 245/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.0185 - val_loss: 0.0184
    Epoch 246/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0180 - val_loss: 0.0181
    Epoch 247/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 248/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 249/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 713us/step - loss: 0.0189 - val_loss: 0.0182
    Epoch 250/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 251/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0181 - val_loss: 0.0183
    Epoch 252/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.0175 - val_loss: 0.0182
    Epoch 253/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 881us/step - loss: 0.0187 - val_loss: 0.0183
    Epoch 254/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0187 - val_loss: 0.0183
    Epoch 255/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.0185 - val_loss: 0.0183
    Epoch 256/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 257/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 724us/step - loss: 0.0180 - val_loss: 0.0182
    Epoch 258/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0179 - val_loss: 0.0183
    Epoch 259/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 260/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 261/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 262/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 675us/step - loss: 0.0177 - val_loss: 0.0185
    Epoch 263/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 685us/step - loss: 0.0184 - val_loss: 0.0183
    Epoch 264/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0186 - val_loss: 0.0182
    Epoch 265/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 673us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 266/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 267/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 696us/step - loss: 0.0190 - val_loss: 0.0182
    Epoch 268/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0183 - val_loss: 0.0181
    Epoch 269/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 676us/step - loss: 0.0185 - val_loss: 0.0182
    Epoch 270/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.0182 - val_loss: 0.0181
    Epoch 271/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 0.0180 - val_loss: 0.0181
    Epoch 272/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 705us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 273/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 274/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 963us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 275/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 714us/step - loss: 0.0176 - val_loss: 0.0184
    Epoch 276/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 277/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 278/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.0175 - val_loss: 0.0184
    Epoch 279/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 280/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 0.0178 - val_loss: 0.0182
    Epoch 281/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 282/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 698us/step - loss: 0.0184 - val_loss: 0.0183
    Epoch 283/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 284/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 285/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0182 - val_loss: 0.0184
    Epoch 286/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 287/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 678us/step - loss: 0.0185 - val_loss: 0.0183
    Epoch 288/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 289/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.0174 - val_loss: 0.0182
    Epoch 290/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 291/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 292/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 688us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 293/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0180 - val_loss: 0.0182
    Epoch 294/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 295/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 296/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.0181 - val_loss: 0.0184
    Epoch 297/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 298/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.0185 - val_loss: 0.0182
    Epoch 299/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 714us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 300/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 712us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 301/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 302/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 303/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 688us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 304/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 689us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 305/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0180 - val_loss: 0.0182
    Epoch 306/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0183 - val_loss: 0.0184
    Epoch 307/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 308/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.0184 - val_loss: 0.0184
    Epoch 309/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 310/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 311/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 0.0183 - val_loss: 0.0183
    Epoch 312/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 0.0185 - val_loss: 0.0182
    Epoch 313/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 716us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 314/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step - loss: 0.0186 - val_loss: 0.0182
    Epoch 315/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 656us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 316/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 678us/step - loss: 0.0176 - val_loss: 0.0181
    Epoch 317/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 695us/step - loss: 0.0175 - val_loss: 0.0182
    Epoch 318/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 663us/step - loss: 0.0183 - val_loss: 0.0181
    Epoch 319/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0181 - val_loss: 0.0181
    Epoch 320/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 712us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 321/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 322/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 678us/step - loss: 0.0178 - val_loss: 0.0182
    Epoch 323/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 667us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 324/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.0177 - val_loss: 0.0183
    Epoch 325/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.0180 - val_loss: 0.0183
    Epoch 326/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 327/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0171 - val_loss: 0.0182
    Epoch 328/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.0184 - val_loss: 0.0181
    Epoch 329/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 330/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 837us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 331/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0178 - val_loss: 0.0183
    Epoch 332/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 724us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 333/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 334/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0181 - val_loss: 0.0181
    Epoch 335/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 336/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 337/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0176 - val_loss: 0.0182
    Epoch 338/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 339/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 679us/step - loss: 0.0183 - val_loss: 0.0182
    Epoch 340/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 679us/step - loss: 0.0174 - val_loss: 0.0181
    Epoch 341/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 685us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 342/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 343/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 661us/step - loss: 0.0180 - val_loss: 0.0182
    Epoch 344/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 688us/step - loss: 0.0186 - val_loss: 0.0183
    Epoch 345/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 346/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 687us/step - loss: 0.0189 - val_loss: 0.0181
    Epoch 347/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 682us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 348/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.0184 - val_loss: 0.0183
    Epoch 349/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.0180 - val_loss: 0.0181
    Epoch 350/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 671us/step - loss: 0.0180 - val_loss: 0.0182
    Epoch 351/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0188 - val_loss: 0.0182
    Epoch 352/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0179 - val_loss: 0.0182
    Epoch 353/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 686us/step - loss: 0.0185 - val_loss: 0.0182
    Epoch 354/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 355/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.0189 - val_loss: 0.0182
    Epoch 356/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.0178 - val_loss: 0.0184
    Epoch 357/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 685us/step - loss: 0.0180 - val_loss: 0.0181
    Epoch 358/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 664us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 359/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 670us/step - loss: 0.0184 - val_loss: 0.0182
    Epoch 360/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 677us/step - loss: 0.0177 - val_loss: 0.0182
    Epoch 361/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0178 - val_loss: 0.0184
    Epoch 362/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 0.0182 - val_loss: 0.0182
    Epoch 363/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 955us/step - loss: 0.0181 - val_loss: 0.0182
    Epoch 364/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0171 - val_loss: 0.0182
    Epoch 365/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 715us/step - loss: 0.0182 - val_loss: 0.0183
    Epoch 366/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0184 - val_loss: 0.0182


The history object that is returned contains a history of the training and validation losses. We can plot this to view how the model trained over the epochs. In the case the validation loss starts to increase our early stopping will kick in and halt the model training.


```python
# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.show()
```


    
![png](Autoencoder_files/Autoencoder_20_0.png)
    


# Assess the model performance

Now the model has finished training we can view its performance by viewing some specific examples. We will create a sinusoidal signal and add some Gaussian noise to it


```python
# noisy sin
total = 10
batch_size = T.size
T_extend = np.linspace(-1, 1, 2514)

complete_sin = np.sin(T_extend * np.pi * 20)
noise_sin = complete_sin + np.random.normal(0, 0.5, T_extend.size)
plt.plot(T_extend[:total*batch_size], noise_sin[:total*batch_size], alpha=0.5)

noisy_sin_batch = np.zeros((total*batch_size, batch_size))
for idx in range(total*batch_size):
    noisy_sin_batch[idx] = noise_sin[idx:(idx+batch_size)]

preds = np.zeros((total*batch_size, batch_size))
for i in range(total*batch_size):
    preds[i] = quant_mod.predict(noisy_sin_batch[i].reshape(1, -1), scale=False, unscale_output=False)

plt.plot(T_extend[:total*batch_size], preds[:, 0])
plt.show()
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step



    
![png](Autoencoder_files/Autoencoder_23_1.png)
    


We can also compare this to a traditional Gaussian smoothing filter to compare the performance of the autoencoder. To achieve this, we convolve the signal with a Gaussian with the appropriate standard deviation with our noisy signal. This is equivalent to passing the signal through a low pass filter. 

As seen in the graphs below, the autoencoder performs comparably to the Gaussian smoothing, showing the effectiveness of the technique. 


```python
# smooth the input using a Gaussian filter
filter = np.exp(-np.linspace(-1, 1, total*batch_size)**2 / 5e-4)
filter /= np.sum(filter)
conv = np.convolve(filter, noise_sin[:total*batch_size], mode='same')

fig, ax = plt.subplots(1, 2, figsize=(15,5), sharey=True)
# plot the smoothed solution
ax[0].plot(T_extend[:total*batch_size], conv[:total*batch_size], label='Gaussian Filter')
ax[0].plot(T_extend[:total*batch_size], noise_sin[:total*batch_size], alpha=0.4, label='Noisy Sin')
ax[0].plot(T_extend[:total*batch_size], complete_sin[:total*batch_size], label='Original Sin')
ax[0].set_title('Smoothed input')
ax[0].set_ylabel('Voltage (V)')
ax[0].set_xlabel('Time (arb.)')
ax[0].legend()

# plot the autoencoder solution
ax[1].plot(T_extend[:total*batch_size], preds[:, 0], label='Autoencoder output')
ax[1].plot(T_extend[:total*batch_size], noise_sin[:total*batch_size], alpha=0.4, label='Noisy Sin')
ax[1].plot(T_extend[:total*batch_size], complete_sin[:total*batch_size], label='Original Sin')
ax[1].set_title('Autoencoder input')
ax[1].set_xlabel('Time (arb.)')
ax[1].legend()

plt.tight_layout()
plt.show()
```


    
![png](Autoencoder_files/Autoencoder_25_0.png)
    


# Deploying the Network onto a Moku:Pro

Now that the network has been trained, we can deploy it onto a Moku:Pro device. By saving the network with 1 input channel and 1 output channel, the network can be configured to take in 32 downsampled points of an input channel, and produce 32 points sequentially through its single output channel. We do this using the `save_linn` function:


```python
save_linn(quant_mod, input_channels=1, output_channels=1, file_name='autoencoder.linn')
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 78 cycles


The sliding window nature of the serial network inputs and outputs means each consecutive batch of 32 outputs the network produces will overlap with 31 pf the previous outputs. To avoid overlapping the same output samples, we want the instrument to only output the most recent data point, which is the value of the last neuron in the last layer. To achieve this, we will use the optional `output_mapping` keyword argument to the `save_linn` function that lets us choose which output neurons are present in the final layer. We will set the `output_mapping` to only contain the final neuron in the last layer. 


```python
save_linn(quant_mod, input_channels=1, output_channels=1, file_name='autoencoder.linn', output_mapping=[T.size-1])
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 78 cycles


With this adjusted model, we can now de-noise some signals on a moku device. This shows an example simulation of producing, de-noising and viewing a periodic signal using a Moku:Pro in multi-instrument mode. 

![](./Autoencoder_Screenshots/MIM_Setup.png "Multi Instrument Mode Setup")

Using multi-instrument mode, we can simulate a real-world signal with variable amounts of noise. We start by configuring a waveform generator to produce a periodic signal and some Gaussian noise. 

![](./Autoencoder_Screenshots/Waveform_Generator_Sine.png "Waveform Generator Producing a Sine Wave and Gaussian Noise")

We can use the control matrix in the PID isntrument to linearly combine the noise and signal, allowing us to significantly increase the amplitude of the noise relative to the underlying signal. Only `Out A` of the PID is used in this example. The red trace shows the incoming noiseless sine wave, and the blue trace shows the signal after noise was added to it. 

![](./Autoencoder_Screenshots/PID_Setup.png "PID Controller Adding a Linear Combination of Two Inputs")

We connect this noisy periodic signal to the input of the neural network instrument, with the adjusted model loaded as our network configuration. 

![](./Autoencoder_Screenshots/Neural_Network_Setup.png "Neural Network Loaded with Autoencoder Configuration")

Finally, we can view the original waveform, noise, noisy waveform, and output of the network using an oscilloscope instrument. The resulting network output works reasonably effectively as a denoised version of the original signal. These four signals are shown in order as the red (original waveform), blue (gaussian noise), green (noisy waveform) and yellow (de-noised signal) traces below. 

![](./Autoencoder_Screenshots/Sine_Results.png "Autoencoder De-Noising Sinusoidal Signal")

This works on a variety of periodic waveforms, including sine waves, triangle waves, and a cardiac waveform as a few examples. The same experiment repeated with triangle and cardiac waveforms is shown below, just requiring a reconfiguration of the waveform generator instrument to produce the desired noiselss signal. 

![](./Autoencoder_Screenshots/Triangle_Results.png "Autoencoder De-Noising Triangular Signal")

![](./Autoencoder_Screenshots/Cardiac_Results.png "Autoencoder De-Noising Cardiac Signal")
