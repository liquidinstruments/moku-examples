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

    2024-10-17 11:11:36.668437: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-17 11:11:36.669260: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2024-10-17 11:11:36.670985: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
    2024-10-17 11:11:36.676123: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-10-17 11:11:36.686189: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-10-17 11:11:36.686200: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-10-17 11:11:36.693237: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-17 11:11:37.114003: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


# Data Generation

To create an effective autoencoder we need to train the model to compress the type of signals that we might expect to see on the Moku. In this example we are simply going to simulate a ringdown signal and apply some noise to it. We will construct a dataset from these noisy signals and allow the autoencoder to “figure out” what the best encoding should be. Since autoencoders are unsupervised learners we don’t need to tell it what the denoised signal should be.

We will start generating our training set by defining the timebase over which we expect to see the signal. We specify a width of 100 to match the maximum width of the Moku neural network input. Note that we define it in terms of 2π so that the frequency of our ringdown is conveniently defined.


```python
# time base, set to width of the network input
T = np.linspace(0, np.pi*2, 100)
```

We define a ringdown function which will give us the signal we expect to see, this is simply a sinusoidal function with an exponentially decaying envelope. The `ring_down()` function will return two arrays for a given decay, frequency and noise. The first array is the signal without noise while the second has the noise added. In general, we may not have access to the signal without noise, however, in this case we will simply use it for illustrative purposes.


```python
# ringdown function that we expect to acquire
def ring_down(T, decay, frequency, noise):
    """
    :param T: Input array of times
    :param decay: the decay constant of the exponential
    :param frequency: the frequency of the ringdown oscillations
    :param noise: size of the noise applied to the signal
    :return: Two arrays for the signal without and with noise
    """
    # pick a phase randomly
    phase = np.random.uniform(0, np.pi*2, 1)[0]

    # get the signal with and without noise
    no_noise = np.exp(-T/decay)*np.sin(frequency*T + phase)
    with_noise = no_noise + np.random.uniform(-noise, noise, len(T))

    return no_noise, with_noise

# generate an example plot
rd_nn, rd_wn = ring_down(T, 0.8, 7, 0.2)
plt.plot(T/(np.pi*2), rd_nn)
plt.plot(T/(np.pi*2), rd_wn)
plt.legend(['True signal', 'Noisy signal'])
plt.xlabel('Time (arb.)')
plt.ylabel('Voltage (V)')
plt.show()
```


    
![png](./Autoencoder_files/Autoencoder_7_0.png)
    


Here we can see that we end up with a noisy version of the ring down signal for a specific frequency and decay. We will now generate a set of training data that we will use to train the model. To do this, we will generate a large number of noisy signals with random frequencies in the range `[3, 15]` and random decays between `[0.8, 1.5]`. We will fix the noise as having amplitude `0.2`. We will save these into an array and also keep track of the noiseless traces for comparison later.


```python
# define the length of our training data
data_len = 1000

# pre-define the arrays
training_data = np.zeros((data_len, T.size))
noiseless = np.zeros((data_len, T.size))

# generate a all of the random waveforms for training and store them
for idx in tqdm(range(data_len)):
    decay = np.random.uniform(0.8, 1.5, 1)[0]
    frequency = np.random.uniform(3, 15, 1)[0]

    # get the ringdown arrays
    Y_no_noise, Y_train = ring_down(T, decay, frequency, 0.2)

    training_data[idx, :] = Y_train
    noiseless[idx, :] = Y_no_noise
```

    100%|██████████| 1000/1000 [00:00<00:00, 112180.16it/s]


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

We now need to define our model structure. To define a structure we will pass a list which contains the model definition. The definition is expected to be a list of tuples of either `(layer_size)` or `(layer_size, layer_activation)`. In our case we are going to define a 3 layer autoencoder which constricts to a latent space size of `8` with the intermediate layers each having a `tanh` activation function. The `show_summary` flag will output the model structure constructed in Tensorflow. There will be a number of intermediate layers for clipping the outputs to conform with the Moku FPGA requirements, but these can largely be ignored. 


```python
# model definition for an autoencoder
model_definition = [(32, 'tanh'), (8, 'tanh'), (32, 'tanh'), (T.size, 'linear')]

# build the model
quant_mod.construct_model(model_definition, show_summary=True)
```

    2024-10-17 11:11:37.510857: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
    2024-10-17 11:11:37.511250: W tensorflow/core/common_runtime/gpu/gpu_device.cc:2251] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">3,232</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">264</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">288</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_2             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">3,300</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_3             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,084</span> (27.67 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,084</span> (27.67 KB)
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
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2661 - val_loss: 0.2386
    Epoch 2/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 839us/step - loss: 0.2293 - val_loss: 0.2215
    Epoch 3/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 837us/step - loss: 0.2153 - val_loss: 0.2100
    Epoch 4/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.2052 - val_loss: 0.2014
    Epoch 5/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 798us/step - loss: 0.1971 - val_loss: 0.1955
    Epoch 6/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 0.1901 - val_loss: 0.1917
    Epoch 7/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.1846 - val_loss: 0.1888
    Epoch 8/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1846 - val_loss: 0.1864
    Epoch 9/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.1814 - val_loss: 0.1846
    Epoch 10/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.1812 - val_loss: 0.1831
    Epoch 11/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.1787 - val_loss: 0.1821
    Epoch 12/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.1763 - val_loss: 0.1809
    Epoch 13/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.1752 - val_loss: 0.1797
    Epoch 14/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.1753 - val_loss: 0.1787
    Epoch 15/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1756 - val_loss: 0.1783
    Epoch 16/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.1742 - val_loss: 0.1777
    Epoch 17/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.1732 - val_loss: 0.1770
    Epoch 18/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.1731 - val_loss: 0.1766
    Epoch 19/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.1734 - val_loss: 0.1764
    Epoch 20/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.1714 - val_loss: 0.1759
    Epoch 21/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1701 - val_loss: 0.1757
    Epoch 22/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.1712 - val_loss: 0.1754
    Epoch 23/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 705us/step - loss: 0.1704 - val_loss: 0.1752
    Epoch 24/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1717 - val_loss: 0.1750
    Epoch 25/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.1709 - val_loss: 0.1748
    Epoch 26/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 714us/step - loss: 0.1699 - val_loss: 0.1746
    Epoch 27/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 724us/step - loss: 0.1704 - val_loss: 0.1746
    Epoch 28/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.1695 - val_loss: 0.1743
    Epoch 29/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.1697 - val_loss: 0.1743
    Epoch 30/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 712us/step - loss: 0.1697 - val_loss: 0.1743
    Epoch 31/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 714us/step - loss: 0.1688 - val_loss: 0.1742
    Epoch 32/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 716us/step - loss: 0.1691 - val_loss: 0.1740
    Epoch 33/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.1704 - val_loss: 0.1739
    Epoch 34/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.1695 - val_loss: 0.1739
    Epoch 35/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.1693 - val_loss: 0.1739
    Epoch 36/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1688 - val_loss: 0.1738
    Epoch 37/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.1693 - val_loss: 0.1738
    Epoch 38/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.1688 - val_loss: 0.1735
    Epoch 39/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 715us/step - loss: 0.1696 - val_loss: 0.1736
    Epoch 40/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.1693 - val_loss: 0.1736
    Epoch 41/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.1699 - val_loss: 0.1734
    Epoch 42/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.1686 - val_loss: 0.1735
    Epoch 43/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.1693 - val_loss: 0.1734
    Epoch 44/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.1682 - val_loss: 0.1735
    Epoch 45/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 716us/step - loss: 0.1691 - val_loss: 0.1735
    Epoch 46/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.1690 - val_loss: 0.1736
    Epoch 47/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 825us/step - loss: 0.1686 - val_loss: 0.1733
    Epoch 48/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 810us/step - loss: 0.1680 - val_loss: 0.1732
    Epoch 49/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1687 - val_loss: 0.1733
    Epoch 50/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1677 - val_loss: 0.1733
    Epoch 51/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.1686 - val_loss: 0.1732
    Epoch 52/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.1674 - val_loss: 0.1733
    Epoch 53/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 700us/step - loss: 0.1684 - val_loss: 0.1733
    Epoch 54/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.1674 - val_loss: 0.1732
    Epoch 55/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1671 - val_loss: 0.1732
    Epoch 56/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.1683 - val_loss: 0.1733
    Epoch 57/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.1681 - val_loss: 0.1730
    Epoch 58/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.1676 - val_loss: 0.1731
    Epoch 59/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.1680 - val_loss: 0.1731
    Epoch 60/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.1680 - val_loss: 0.1731
    Epoch 61/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 872us/step - loss: 0.1684 - val_loss: 0.1731
    Epoch 62/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.1673 - val_loss: 0.1731
    Epoch 63/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.1665 - val_loss: 0.1731
    Epoch 64/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 701us/step - loss: 0.1687 - val_loss: 0.1732
    Epoch 65/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.1684 - val_loss: 0.1731
    Epoch 66/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 0.1678 - val_loss: 0.1730
    Epoch 67/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 720us/step - loss: 0.1672 - val_loss: 0.1730
    Epoch 68/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.1672 - val_loss: 0.1730
    Epoch 69/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 723us/step - loss: 0.1672 - val_loss: 0.1731
    Epoch 70/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 812us/step - loss: 0.1675 - val_loss: 0.1730
    Epoch 71/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.1682 - val_loss: 0.1730
    Epoch 72/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.1674 - val_loss: 0.1729
    Epoch 73/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.1663 - val_loss: 0.1729
    Epoch 74/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1675 - val_loss: 0.1729
    Epoch 75/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.1665 - val_loss: 0.1728
    Epoch 76/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1663 - val_loss: 0.1728
    Epoch 77/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 0.1668 - val_loss: 0.1729
    Epoch 78/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.1675 - val_loss: 0.1729
    Epoch 79/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1681 - val_loss: 0.1729
    Epoch 80/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1675 - val_loss: 0.1729
    Epoch 81/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step - loss: 0.1683 - val_loss: 0.1730
    Epoch 82/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 0.1673 - val_loss: 0.1731
    Epoch 83/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.1666 - val_loss: 0.1730
    Epoch 84/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.1690 - val_loss: 0.1729
    Epoch 85/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.1669 - val_loss: 0.1728
    Epoch 86/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1672 - val_loss: 0.1726
    Epoch 87/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.1677 - val_loss: 0.1725
    Epoch 88/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 694us/step - loss: 0.1658 - val_loss: 0.1726
    Epoch 89/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1665 - val_loss: 0.1727
    Epoch 90/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.1674 - val_loss: 0.1726
    Epoch 91/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.1683 - val_loss: 0.1727
    Epoch 92/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1664 - val_loss: 0.1726
    Epoch 93/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.1671 - val_loss: 0.1727
    Epoch 94/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.1670 - val_loss: 0.1728
    Epoch 95/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.1674 - val_loss: 0.1726
    Epoch 96/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1664 - val_loss: 0.1727
    Epoch 97/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.1671 - val_loss: 0.1727
    Epoch 98/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 684us/step - loss: 0.1670 - val_loss: 0.1727
    Epoch 99/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 691us/step - loss: 0.1659 - val_loss: 0.1729
    Epoch 100/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.1664 - val_loss: 0.1726
    Epoch 101/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 703us/step - loss: 0.1665 - val_loss: 0.1727
    Epoch 102/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.1658 - val_loss: 0.1726
    Epoch 103/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.1666 - val_loss: 0.1727
    Epoch 104/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1662 - val_loss: 0.1726
    Epoch 105/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1657 - val_loss: 0.1725
    Epoch 106/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 713us/step - loss: 0.1655 - val_loss: 0.1723
    Epoch 107/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 636us/step - loss: 0.1669 - val_loss: 0.1724
    Epoch 108/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 659us/step - loss: 0.1667 - val_loss: 0.1725
    Epoch 109/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 679us/step - loss: 0.1658 - val_loss: 0.1724
    Epoch 110/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 673us/step - loss: 0.1663 - val_loss: 0.1723
    Epoch 111/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.1651 - val_loss: 0.1723
    Epoch 112/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.1653 - val_loss: 0.1724
    Epoch 113/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1661 - val_loss: 0.1723
    Epoch 114/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 715us/step - loss: 0.1650 - val_loss: 0.1721
    Epoch 115/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1663 - val_loss: 0.1720
    Epoch 116/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.1647 - val_loss: 0.1722
    Epoch 117/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 693us/step - loss: 0.1664 - val_loss: 0.1722
    Epoch 118/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1667 - val_loss: 0.1722
    Epoch 119/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.1656 - val_loss: 0.1719
    Epoch 120/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.1653 - val_loss: 0.1718
    Epoch 121/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.1639 - val_loss: 0.1720
    Epoch 122/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 668us/step - loss: 0.1644 - val_loss: 0.1718
    Epoch 123/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 638us/step - loss: 0.1637 - val_loss: 0.1717
    Epoch 124/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.1640 - val_loss: 0.1716
    Epoch 125/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 662us/step - loss: 0.1660 - val_loss: 0.1718
    Epoch 126/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 0.1638 - val_loss: 0.1718
    Epoch 127/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.1642 - val_loss: 0.1715
    Epoch 128/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 724us/step - loss: 0.1652 - val_loss: 0.1715
    Epoch 129/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.1637 - val_loss: 0.1716
    Epoch 130/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.1638 - val_loss: 0.1713
    Epoch 131/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 0.1641 - val_loss: 0.1714
    Epoch 132/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.1635 - val_loss: 0.1715
    Epoch 133/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 0.1634 - val_loss: 0.1711
    Epoch 134/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 0.1646 - val_loss: 0.1712
    Epoch 135/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 887us/step - loss: 0.1622 - val_loss: 0.1713
    Epoch 136/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.1635 - val_loss: 0.1711
    Epoch 137/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.1646 - val_loss: 0.1711
    Epoch 138/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.1636 - val_loss: 0.1711
    Epoch 139/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 719us/step - loss: 0.1623 - val_loss: 0.1709
    Epoch 140/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 723us/step - loss: 0.1632 - val_loss: 0.1711
    Epoch 141/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 0.1624 - val_loss: 0.1711
    Epoch 142/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 698us/step - loss: 0.1623 - val_loss: 0.1711
    Epoch 143/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.1626 - val_loss: 0.1710
    Epoch 144/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1620 - val_loss: 0.1708
    Epoch 145/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.1624 - val_loss: 0.1710
    Epoch 146/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.1628 - val_loss: 0.1708
    Epoch 147/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1642 - val_loss: 0.1708
    Epoch 148/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1619 - val_loss: 0.1705
    Epoch 149/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 717us/step - loss: 0.1631 - val_loss: 0.1704
    Epoch 150/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.1624 - val_loss: 0.1704
    Epoch 151/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 695us/step - loss: 0.1608 - val_loss: 0.1705
    Epoch 152/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 693us/step - loss: 0.1610 - val_loss: 0.1704
    Epoch 153/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1612 - val_loss: 0.1700
    Epoch 154/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 712us/step - loss: 0.1627 - val_loss: 0.1702
    Epoch 155/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.1616 - val_loss: 0.1703
    Epoch 156/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.1624 - val_loss: 0.1701
    Epoch 157/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 846us/step - loss: 0.1612 - val_loss: 0.1701
    Epoch 158/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.1617 - val_loss: 0.1703
    Epoch 159/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 706us/step - loss: 0.1615 - val_loss: 0.1701
    Epoch 160/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.1616 - val_loss: 0.1700
    Epoch 161/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1595 - val_loss: 0.1699
    Epoch 162/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.1612 - val_loss: 0.1700
    Epoch 163/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.1606 - val_loss: 0.1699
    Epoch 164/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1612 - val_loss: 0.1699
    Epoch 165/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1612 - val_loss: 0.1697
    Epoch 166/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.1611 - val_loss: 0.1698
    Epoch 167/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.1597 - val_loss: 0.1696
    Epoch 168/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1602 - val_loss: 0.1696
    Epoch 169/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1595 - val_loss: 0.1698
    Epoch 170/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 721us/step - loss: 0.1589 - val_loss: 0.1698
    Epoch 171/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 707us/step - loss: 0.1605 - val_loss: 0.1695
    Epoch 172/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 704us/step - loss: 0.1607 - val_loss: 0.1695
    Epoch 173/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1597 - val_loss: 0.1695
    Epoch 174/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 710us/step - loss: 0.1600 - val_loss: 0.1697
    Epoch 175/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 697us/step - loss: 0.1583 - val_loss: 0.1697
    Epoch 176/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 690us/step - loss: 0.1584 - val_loss: 0.1697
    Epoch 177/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 711us/step - loss: 0.1591 - val_loss: 0.1696
    Epoch 178/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.1596 - val_loss: 0.1694
    Epoch 179/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 702us/step - loss: 0.1595 - val_loss: 0.1695
    Epoch 180/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 703us/step - loss: 0.1593 - val_loss: 0.1692
    Epoch 181/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 699us/step - loss: 0.1592 - val_loss: 0.1696
    Epoch 182/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 717us/step - loss: 0.1587 - val_loss: 0.1695
    Epoch 183/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 853us/step - loss: 0.1586 - val_loss: 0.1693
    Epoch 184/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.1583 - val_loss: 0.1694
    Epoch 185/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 703us/step - loss: 0.1584 - val_loss: 0.1693
    Epoch 186/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.1587 - val_loss: 0.1693
    Epoch 187/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.1584 - val_loss: 0.1695
    Epoch 188/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.1581 - val_loss: 0.1694
    Epoch 189/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 708us/step - loss: 0.1584 - val_loss: 0.1693
    Epoch 190/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1584 - val_loss: 0.1694
    Epoch 191/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 727us/step - loss: 0.1576 - val_loss: 0.1695
    Epoch 192/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.1581 - val_loss: 0.1691
    Epoch 193/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1581 - val_loss: 0.1693
    Epoch 194/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.1580 - val_loss: 0.1691
    Epoch 195/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.1572 - val_loss: 0.1690
    Epoch 196/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.1586 - val_loss: 0.1692
    Epoch 197/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.1580 - val_loss: 0.1694
    Epoch 198/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.1560 - val_loss: 0.1693
    Epoch 199/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.1574 - val_loss: 0.1691
    Epoch 200/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1567 - val_loss: 0.1693
    Epoch 201/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1579 - val_loss: 0.1692
    Epoch 202/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1579 - val_loss: 0.1692
    Epoch 203/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.1567 - val_loss: 0.1691
    Epoch 204/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1571 - val_loss: 0.1694
    Epoch 205/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.1558 - val_loss: 0.1693
    Epoch 206/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 689us/step - loss: 0.1572 - val_loss: 0.1691
    Epoch 207/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.1571 - val_loss: 0.1690
    Epoch 208/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 654us/step - loss: 0.1569 - val_loss: 0.1689
    Epoch 209/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.1565 - val_loss: 0.1691
    Epoch 210/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 950us/step - loss: 0.1564 - val_loss: 0.1690
    Epoch 211/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 871us/step - loss: 0.1561 - val_loss: 0.1688
    Epoch 212/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.1564 - val_loss: 0.1691
    Epoch 213/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.1560 - val_loss: 0.1692
    Epoch 214/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.1563 - val_loss: 0.1693
    Epoch 215/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.1566 - val_loss: 0.1690
    Epoch 216/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.1553 - val_loss: 0.1691
    Epoch 217/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 633us/step - loss: 0.1560 - val_loss: 0.1690
    Epoch 218/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 651us/step - loss: 0.1556 - val_loss: 0.1691
    Epoch 219/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 618us/step - loss: 0.1552 - val_loss: 0.1690
    Epoch 220/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 628us/step - loss: 0.1564 - val_loss: 0.1688
    Epoch 221/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 664us/step - loss: 0.1557 - val_loss: 0.1690
    Epoch 222/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 705us/step - loss: 0.1555 - val_loss: 0.1690
    Epoch 223/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 683us/step - loss: 0.1557 - val_loss: 0.1690
    Epoch 224/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.1555 - val_loss: 0.1692
    Epoch 225/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1550 - val_loss: 0.1690
    Epoch 226/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.1557 - val_loss: 0.1691
    Epoch 227/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.1548 - val_loss: 0.1693
    Epoch 228/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.1558 - val_loss: 0.1691
    Epoch 229/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.1550 - val_loss: 0.1691
    Epoch 230/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.1554 - val_loss: 0.1691
    Epoch 231/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.1545 - val_loss: 0.1693
    Epoch 232/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.1543 - val_loss: 0.1692
    Epoch 233/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.1548 - val_loss: 0.1691
    Epoch 234/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 0.1543 - val_loss: 0.1690
    Epoch 235/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 830us/step - loss: 0.1541 - val_loss: 0.1689
    Epoch 236/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 829us/step - loss: 0.1540 - val_loss: 0.1690
    Epoch 237/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.1547 - val_loss: 0.1691
    Epoch 238/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 800us/step - loss: 0.1550 - val_loss: 0.1694
    Epoch 239/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.1542 - val_loss: 0.1691
    Epoch 240/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.1544 - val_loss: 0.1692
    Epoch 241/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.1547 - val_loss: 0.1691
    Epoch 242/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.1544 - val_loss: 0.1693
    Epoch 243/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.1538 - val_loss: 0.1692
    Epoch 244/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.1540 - val_loss: 0.1691
    Epoch 245/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.1543 - val_loss: 0.1693
    Epoch 246/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.1540 - val_loss: 0.1696
    Epoch 247/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 692us/step - loss: 0.1541 - val_loss: 0.1693
    Epoch 248/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.1536 - val_loss: 0.1695
    Epoch 249/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.1541 - val_loss: 0.1694
    Epoch 250/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 787us/step - loss: 0.1533 - val_loss: 0.1694
    Epoch 251/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.1546 - val_loss: 0.1693
    Epoch 252/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.1539 - val_loss: 0.1692
    Epoch 253/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 725us/step - loss: 0.1549 - val_loss: 0.1695
    Epoch 254/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 715us/step - loss: 0.1538 - val_loss: 0.1695
    Epoch 255/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.1536 - val_loss: 0.1695
    Epoch 256/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 881us/step - loss: 0.1537 - val_loss: 0.1695
    Epoch 257/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.1534 - val_loss: 0.1694
    Epoch 258/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 0.1546 - val_loss: 0.1692
    Epoch 259/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.1528 - val_loss: 0.1693
    Epoch 260/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.1535 - val_loss: 0.1696
    Epoch 261/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.1533 - val_loss: 0.1698


The history object that is returned contains a history of the training and validation losses. We can plot this to view how the model trained over the epochs. In the case the validation loss starts to increase our early stopping will kick in and halt the model training.


```python
# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.show()
```


    
![png](./Autoencoder_files/Autoencoder_20_0.png)
    


# Assess the model performance

Now the model has finished training we can view its performance by viewing some specific examples. We will pick an example from the validation set (the last 10%) to ensure our model is behaving correctly. Calling the predict method will effectively give use the denoised signal via the autoencoder.


```python
# get all the denoised signals
preds = quant_mod.predict(training_data)

# specify some signal to look at
N = 990

fig, ax = plt.subplots(1, 2, figsize=(15,5), sharey=True)
# the noisy signal
ax[0].plot(T/(np.pi*2), training_data[N])
ax[0].set_title('Noisy input')
ax[0].set_ylabel('Voltage (V)')
ax[0].set_xlabel('Time (arb.)')

# the recovered signal
ax[1].plot(T/(np.pi*2), preds[N])
ax[1].set_title('Denoised and real signal')
ax[1].plot(T/(np.pi*2), noiseless[N])
ax[1].set_xlabel('Time (arb.)')
ax[1].legend(['Denoised', 'Real'])
plt.tight_layout()
plt.show()
```

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 840us/step



    
![png](./Autoencoder_files/Autoencoder_23_1.png)
    


We can also compare this to a traditional Gaussian smoothing filter to compare the performance of the autoencoder.


```python
# smooth the input using a Gaussian filter
filter = np.exp(-np.linspace(-1, 1, T.size)**2 / 5e-4)
filter /= np.sum(filter)
conv = np.convolve(filter, training_data[N], mode='same')

fig, ax = plt.subplots(1, 2, figsize=(15,5), sharey=True)
# plot the smoothed solution
ax[0].plot(T/(np.pi*2), noiseless[N])
ax[0].plot(T/(np.pi*2), conv)
ax[0].set_title('Smoothed input')
ax[0].set_ylabel('Voltage (V)')
ax[0].set_xlabel('Time (arb.)')

# plot the autoencoder solution
ax[1].plot(T/(np.pi*2), noiseless[N])
ax[1].plot(T/(np.pi*2), preds[N])
ax[1].set_title('Autoencoder input')
ax[1].set_xlabel('Time (arb.)')

plt.tight_layout()
plt.show()
```


    
![png](./Autoencoder_files/Autoencoder_25_0.png)
    


# Deploying the Network onto a Moku:Pro

Now that the network has been trained, we can deploy it onto a Moku:Pro device. By saving the network with 1 input channel and 1 output channel, the network can be configured to take in 100 downsampled points of an input channel, and produce 100 points sequentially through its single output channel. We do this using the `save_linn` function:


```python
save_linn(quant_mod, input_channels=1, output_channels=1, file_name='autoencoder.linn')
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 184 cycles


The sliding window nature of the serial network inputs and outputs means each consecutive batch of 100 outputs the network produces will overlap with the previous 99 outputs. To avoid overlapping the same output samples, we want the instrument to only output the most recent data point, which is the value of the last neuron in the last layer. To achieve this, we will use the optional `output_mapping` keyword argument to the `save_linn` function that lets us choose which output neurons are present in the final layer. We will set the `output_mapping` to only contain the final neuron in the last layer. 


```python
save_linn(quant_mod, input_channels=1, output_channels=1, file_name='autoencoder_one_output.linn', output_mapping=[T.size-1])
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 184 cycles


With this adjusted model, we can now de-noise some signals on a moku device. This shows an example simulation of producing, de-noising and viewing a periodic signal using a Moku:Pro in multi-instrument mode. 

![](./Autoencoder_files/MIM_Setup.png "Multi Instrument Mode Setup")

Using multi-instrument mode, we can simulate a real-world signal with variable amounts of noise. We start by configuring a waveform generator to produce a periodic signal and some Gaussian noise. 

![](./Autoencoder_files/Waveform_Generator_Sine.png "Waveform Generator Producing a Sine Wave and Gaussian Noise")

We can use the control matrix in the PID isntrument to linearly combine the noise and signal, allowing us to significantly increase the amplitude of the noise relative to the underlying signal. Only `Out A` of the PID is used in this example. The red trace shows the incoming noiseless sine wave, and the blue trace shows the signal after noise was added to it. 

![](./Autoencoder_files/PID_Setup.png "PID Controller Adding a Linear Combination of Two Inputs")

We connect this noisy periodic signal to the input of the neural network instrument, with the adjusted model loaded as our network configuration. 

![](./Autoencoder_files/Neural_Network_Setup.png "Neural Network Loaded with Autoencoder Configuration")

Finally, we can view the original waveform, noise, noisy waveform, and output of the network using an oscilloscope instrument. The resulting network output works reasonably effectively as a denoised version of the original signal. These four signals are shown in order as the red (original waveform), blue (gaussian noise), green (noisy waveform) and yellow (de-noised signal) traces below. 

![](./Autoencoder_files/Sine_Results.png "Autoencoder De-Noising Sinusoidal Signal")

This works on a variety of periodic waveforms, including sine waves, triangle waves, and a cardiac waveform as a few examples. The same experiment repeated with triangle and cardiac waveforms is shown below, just requiring a reconfiguration of the waveform generator instrument to produce the desired noiselss signal. 

![](./Autoencoder_files/Triangle_Results.png "Autoencoder De-Noising Triangular Signal")

![](./Autoencoder_files/Cardiac_Results.png "Autoencoder De-Noising Cardiac Signal")
