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
# define the lengtht of our training data
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

    100%|██████████| 1000/1000 [00:00<00:00, 76675.51it/s]


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
model_definition = [(32, 'tanh'), (8, 'tanh'), (32, 'tanh'), (100, 'linear')]

# build the model
quant_mod.construct_model(model_definition, show_summary=True)
```


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
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 4ms/step - loss: 0.2618 - val_loss: 0.2320
    Epoch 2/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2289 - val_loss: 0.2172
    Epoch 3/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2142 - val_loss: 0.2074
    Epoch 4/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 993us/step - loss: 0.2049 - val_loss: 0.1996
    Epoch 5/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 967us/step - loss: 0.1977 - val_loss: 0.1934
    Epoch 6/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 933us/step - loss: 0.1913 - val_loss: 0.1885
    Epoch 7/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 867us/step - loss: 0.1859 - val_loss: 0.1853
    Epoch 8/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 893us/step - loss: 0.1816 - val_loss: 0.1827
    Epoch 9/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 928us/step - loss: 0.1795 - val_loss: 0.1812
    Epoch 10/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 986us/step - loss: 0.1782 - val_loss: 0.1799
    Epoch 11/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 907us/step - loss: 0.1761 - val_loss: 0.1790
    Epoch 12/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 985us/step - loss: 0.1755 - val_loss: 0.1786
    Epoch 13/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 980us/step - loss: 0.1749 - val_loss: 0.1779
    Epoch 14/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1743 - val_loss: 0.1775
    Epoch 15/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1738 - val_loss: 0.1769
    Epoch 16/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 949us/step - loss: 0.1728 - val_loss: 0.1765
    Epoch 17/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 0.1723 - val_loss: 0.1762
    Epoch 18/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 887us/step - loss: 0.1730 - val_loss: 0.1760
    Epoch 19/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 877us/step - loss: 0.1714 - val_loss: 0.1758
    Epoch 20/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 946us/step - loss: 0.1707 - val_loss: 0.1755
    Epoch 21/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 932us/step - loss: 0.1703 - val_loss: 0.1753
    Epoch 22/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 0.1703 - val_loss: 0.1750
    Epoch 23/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 880us/step - loss: 0.1707 - val_loss: 0.1748
    Epoch 24/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 917us/step - loss: 0.1706 - val_loss: 0.1746
    Epoch 25/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 893us/step - loss: 0.1693 - val_loss: 0.1746
    Epoch 26/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 889us/step - loss: 0.1701 - val_loss: 0.1744
    Epoch 27/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 939us/step - loss: 0.1687 - val_loss: 0.1743
    Epoch 28/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 959us/step - loss: 0.1676 - val_loss: 0.1741
    Epoch 29/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 961us/step - loss: 0.1686 - val_loss: 0.1739
    Epoch 30/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 919us/step - loss: 0.1694 - val_loss: 0.1739
    Epoch 31/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1690 - val_loss: 0.1738
    Epoch 32/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 889us/step - loss: 0.1698 - val_loss: 0.1737
    Epoch 33/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 898us/step - loss: 0.1682 - val_loss: 0.1737
    Epoch 34/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 928us/step - loss: 0.1684 - val_loss: 0.1737
    Epoch 35/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 877us/step - loss: 0.1699 - val_loss: 0.1736
    Epoch 36/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 931us/step - loss: 0.1673 - val_loss: 0.1736
    Epoch 37/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.1703 - val_loss: 0.1735
    Epoch 38/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 900us/step - loss: 0.1686 - val_loss: 0.1734
    Epoch 39/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 920us/step - loss: 0.1685 - val_loss: 0.1734
    Epoch 40/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 902us/step - loss: 0.1679 - val_loss: 0.1733
    Epoch 41/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 912us/step - loss: 0.1675 - val_loss: 0.1732
    Epoch 42/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1675 - val_loss: 0.1732
    Epoch 43/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1685 - val_loss: 0.1733
    Epoch 44/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1693 - val_loss: 0.1732
    Epoch 45/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 958us/step - loss: 0.1683 - val_loss: 0.1731
    Epoch 46/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1682 - val_loss: 0.1731
    Epoch 47/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 947us/step - loss: 0.1690 - val_loss: 0.1732
    Epoch 48/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 947us/step - loss: 0.1680 - val_loss: 0.1732
    Epoch 49/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 938us/step - loss: 0.1685 - val_loss: 0.1733
    Epoch 50/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1686 - val_loss: 0.1732
    Epoch 51/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 880us/step - loss: 0.1685 - val_loss: 0.1731
    Epoch 52/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 913us/step - loss: 0.1678 - val_loss: 0.1733
    Epoch 53/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step - loss: 0.1682 - val_loss: 0.1733
    Epoch 54/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 880us/step - loss: 0.1688 - val_loss: 0.1732
    Epoch 55/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 915us/step - loss: 0.1684 - val_loss: 0.1733
    Epoch 56/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 926us/step - loss: 0.1690 - val_loss: 0.1732
    Epoch 57/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1686 - val_loss: 0.1731
    Epoch 58/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1682 - val_loss: 0.1730
    Epoch 59/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 957us/step - loss: 0.1686 - val_loss: 0.1730
    Epoch 60/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 902us/step - loss: 0.1682 - val_loss: 0.1732
    Epoch 61/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 902us/step - loss: 0.1672 - val_loss: 0.1731
    Epoch 62/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 938us/step - loss: 0.1679 - val_loss: 0.1729
    Epoch 63/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 963us/step - loss: 0.1676 - val_loss: 0.1731
    Epoch 64/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 979us/step - loss: 0.1673 - val_loss: 0.1730
    Epoch 65/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1684 - val_loss: 0.1730
    Epoch 66/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 886us/step - loss: 0.1668 - val_loss: 0.1730
    Epoch 67/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 928us/step - loss: 0.1666 - val_loss: 0.1729
    Epoch 68/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 931us/step - loss: 0.1671 - val_loss: 0.1730
    Epoch 69/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 923us/step - loss: 0.1670 - val_loss: 0.1731
    Epoch 70/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1671 - val_loss: 0.1730
    Epoch 71/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 999us/step - loss: 0.1676 - val_loss: 0.1728
    Epoch 72/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 935us/step - loss: 0.1680 - val_loss: 0.1728
    Epoch 73/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 926us/step - loss: 0.1658 - val_loss: 0.1727
    Epoch 74/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 893us/step - loss: 0.1664 - val_loss: 0.1728
    Epoch 75/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 900us/step - loss: 0.1677 - val_loss: 0.1728
    Epoch 76/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 939us/step - loss: 0.1669 - val_loss: 0.1729
    Epoch 77/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1671 - val_loss: 0.1729
    Epoch 78/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 883us/step - loss: 0.1674 - val_loss: 0.1728
    Epoch 79/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 891us/step - loss: 0.1665 - val_loss: 0.1728
    Epoch 80/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1677 - val_loss: 0.1727
    Epoch 81/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 919us/step - loss: 0.1669 - val_loss: 0.1727
    Epoch 82/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 946us/step - loss: 0.1659 - val_loss: 0.1725
    Epoch 83/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 907us/step - loss: 0.1652 - val_loss: 0.1725
    Epoch 84/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1661 - val_loss: 0.1726
    Epoch 85/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 936us/step - loss: 0.1674 - val_loss: 0.1724
    Epoch 86/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.1653 - val_loss: 0.1726
    Epoch 87/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 914us/step - loss: 0.1663 - val_loss: 0.1724
    Epoch 88/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 921us/step - loss: 0.1663 - val_loss: 0.1724
    Epoch 89/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 922us/step - loss: 0.1658 - val_loss: 0.1722
    Epoch 90/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.1664 - val_loss: 0.1721
    Epoch 91/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 0.1666 - val_loss: 0.1720
    Epoch 92/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1664 - val_loss: 0.1719
    Epoch 93/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1662 - val_loss: 0.1719
    Epoch 94/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.1651 - val_loss: 0.1718
    Epoch 95/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1658 - val_loss: 0.1719
    Epoch 96/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 930us/step - loss: 0.1661 - val_loss: 0.1718
    Epoch 97/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step - loss: 0.1660 - val_loss: 0.1715
    Epoch 98/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 881us/step - loss: 0.1653 - val_loss: 0.1716
    Epoch 99/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 913us/step - loss: 0.1656 - val_loss: 0.1712
    Epoch 100/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 934us/step - loss: 0.1652 - val_loss: 0.1712
    Epoch 101/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step - loss: 0.1643 - val_loss: 0.1713
    Epoch 102/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1652 - val_loss: 0.1712
    Epoch 103/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 883us/step - loss: 0.1642 - val_loss: 0.1712
    Epoch 104/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 899us/step - loss: 0.1650 - val_loss: 0.1709
    Epoch 105/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1644 - val_loss: 0.1709
    Epoch 106/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 933us/step - loss: 0.1649 - val_loss: 0.1709
    Epoch 107/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 913us/step - loss: 0.1657 - val_loss: 0.1708
    Epoch 108/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1643 - val_loss: 0.1709
    Epoch 109/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 912us/step - loss: 0.1643 - val_loss: 0.1707
    Epoch 110/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1651 - val_loss: 0.1707
    Epoch 111/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 932us/step - loss: 0.1641 - val_loss: 0.1707
    Epoch 112/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 0.1642 - val_loss: 0.1706
    Epoch 113/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 935us/step - loss: 0.1637 - val_loss: 0.1705
    Epoch 114/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 941us/step - loss: 0.1630 - val_loss: 0.1704
    Epoch 115/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step - loss: 0.1643 - val_loss: 0.1703
    Epoch 116/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 874us/step - loss: 0.1636 - val_loss: 0.1704
    Epoch 117/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1629 - val_loss: 0.1703
    Epoch 118/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 919us/step - loss: 0.1632 - val_loss: 0.1702
    Epoch 119/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1641 - val_loss: 0.1699
    Epoch 120/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 922us/step - loss: 0.1622 - val_loss: 0.1699
    Epoch 121/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 917us/step - loss: 0.1628 - val_loss: 0.1698
    Epoch 122/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 898us/step - loss: 0.1635 - val_loss: 0.1699
    Epoch 123/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 923us/step - loss: 0.1622 - val_loss: 0.1698
    Epoch 124/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 891us/step - loss: 0.1624 - val_loss: 0.1697
    Epoch 125/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 899us/step - loss: 0.1631 - val_loss: 0.1699
    Epoch 126/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 927us/step - loss: 0.1618 - val_loss: 0.1697
    Epoch 127/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 925us/step - loss: 0.1624 - val_loss: 0.1696
    Epoch 128/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 935us/step - loss: 0.1629 - val_loss: 0.1695
    Epoch 129/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 892us/step - loss: 0.1626 - val_loss: 0.1696
    Epoch 130/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 914us/step - loss: 0.1612 - val_loss: 0.1695
    Epoch 131/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 929us/step - loss: 0.1631 - val_loss: 0.1693
    Epoch 132/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 889us/step - loss: 0.1619 - val_loss: 0.1694
    Epoch 133/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 0.1616 - val_loss: 0.1693
    Epoch 134/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 937us/step - loss: 0.1614 - val_loss: 0.1690
    Epoch 135/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1612 - val_loss: 0.1690
    Epoch 136/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step - loss: 0.1616 - val_loss: 0.1692
    Epoch 137/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1617 - val_loss: 0.1691
    Epoch 138/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1613 - val_loss: 0.1690
    Epoch 139/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 900us/step - loss: 0.1612 - val_loss: 0.1690
    Epoch 140/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.1610 - val_loss: 0.1689
    Epoch 141/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 880us/step - loss: 0.1620 - val_loss: 0.1689
    Epoch 142/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 928us/step - loss: 0.1606 - val_loss: 0.1690
    Epoch 143/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step - loss: 0.1607 - val_loss: 0.1690
    Epoch 144/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 924us/step - loss: 0.1608 - val_loss: 0.1688
    Epoch 145/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 914us/step - loss: 0.1608 - val_loss: 0.1688
    Epoch 146/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 950us/step - loss: 0.1609 - val_loss: 0.1687
    Epoch 147/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 945us/step - loss: 0.1609 - val_loss: 0.1688
    Epoch 148/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 903us/step - loss: 0.1589 - val_loss: 0.1687
    Epoch 149/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 895us/step - loss: 0.1591 - val_loss: 0.1686
    Epoch 150/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 933us/step - loss: 0.1604 - val_loss: 0.1684
    Epoch 151/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 940us/step - loss: 0.1596 - val_loss: 0.1684
    Epoch 152/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 925us/step - loss: 0.1603 - val_loss: 0.1684
    Epoch 153/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1602 - val_loss: 0.1685
    Epoch 154/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 889us/step - loss: 0.1590 - val_loss: 0.1685
    Epoch 155/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.1594 - val_loss: 0.1686
    Epoch 156/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 906us/step - loss: 0.1594 - val_loss: 0.1685
    Epoch 157/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 899us/step - loss: 0.1589 - val_loss: 0.1687
    Epoch 158/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.1590 - val_loss: 0.1686
    Epoch 159/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1602 - val_loss: 0.1687
    Epoch 160/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 919us/step - loss: 0.1593 - val_loss: 0.1687
    Epoch 161/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 888us/step - loss: 0.1599 - val_loss: 0.1686
    Epoch 162/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 907us/step - loss: 0.1580 - val_loss: 0.1686
    Epoch 163/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 897us/step - loss: 0.1590 - val_loss: 0.1685
    Epoch 164/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 915us/step - loss: 0.1592 - val_loss: 0.1684
    Epoch 165/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 912us/step - loss: 0.1595 - val_loss: 0.1684
    Epoch 166/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 915us/step - loss: 0.1583 - val_loss: 0.1686
    Epoch 167/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 927us/step - loss: 0.1593 - val_loss: 0.1683
    Epoch 168/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 924us/step - loss: 0.1588 - val_loss: 0.1684
    Epoch 169/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1578 - val_loss: 0.1683
    Epoch 170/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1578 - val_loss: 0.1682
    Epoch 171/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 906us/step - loss: 0.1584 - val_loss: 0.1684
    Epoch 172/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1587 - val_loss: 0.1683
    Epoch 173/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 930us/step - loss: 0.1585 - val_loss: 0.1683
    Epoch 174/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 0.1590 - val_loss: 0.1683
    Epoch 175/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1573 - val_loss: 0.1683
    Epoch 176/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1586 - val_loss: 0.1683
    Epoch 177/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 931us/step - loss: 0.1586 - val_loss: 0.1682
    Epoch 178/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 905us/step - loss: 0.1586 - val_loss: 0.1684
    Epoch 179/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1583 - val_loss: 0.1681
    Epoch 180/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1575 - val_loss: 0.1679
    Epoch 181/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 900us/step - loss: 0.1574 - val_loss: 0.1682
    Epoch 182/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 915us/step - loss: 0.1569 - val_loss: 0.1682
    Epoch 183/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 913us/step - loss: 0.1578 - val_loss: 0.1683
    Epoch 184/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1569 - val_loss: 0.1680
    Epoch 185/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 896us/step - loss: 0.1576 - val_loss: 0.1682
    Epoch 186/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 886us/step - loss: 0.1568 - val_loss: 0.1683
    Epoch 187/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 907us/step - loss: 0.1569 - val_loss: 0.1681
    Epoch 188/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 906us/step - loss: 0.1570 - val_loss: 0.1682
    Epoch 189/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 904us/step - loss: 0.1563 - val_loss: 0.1682
    Epoch 190/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 925us/step - loss: 0.1569 - val_loss: 0.1681
    Epoch 191/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 884us/step - loss: 0.1569 - val_loss: 0.1684
    Epoch 192/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 925us/step - loss: 0.1563 - val_loss: 0.1682
    Epoch 193/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step - loss: 0.1574 - val_loss: 0.1681
    Epoch 194/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 917us/step - loss: 0.1576 - val_loss: 0.1682
    Epoch 195/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 894us/step - loss: 0.1564 - val_loss: 0.1685
    Epoch 196/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 887us/step - loss: 0.1567 - val_loss: 0.1681
    Epoch 197/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 886us/step - loss: 0.1564 - val_loss: 0.1680
    Epoch 198/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 921us/step - loss: 0.1562 - val_loss: 0.1680
    Epoch 199/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1563 - val_loss: 0.1681
    Epoch 200/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 951us/step - loss: 0.1559 - val_loss: 0.1682
    Epoch 201/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 921us/step - loss: 0.1565 - val_loss: 0.1684
    Epoch 202/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 926us/step - loss: 0.1562 - val_loss: 0.1683
    Epoch 203/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.1560 - val_loss: 0.1681
    Epoch 204/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 927us/step - loss: 0.1559 - val_loss: 0.1681
    Epoch 205/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 896us/step - loss: 0.1558 - val_loss: 0.1683
    Epoch 206/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 902us/step - loss: 0.1563 - val_loss: 0.1682
    Epoch 207/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 922us/step - loss: 0.1550 - val_loss: 0.1683
    Epoch 208/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 908us/step - loss: 0.1563 - val_loss: 0.1684
    Epoch 209/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step - loss: 0.1550 - val_loss: 0.1685
    Epoch 210/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 903us/step - loss: 0.1556 - val_loss: 0.1683
    Epoch 211/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 916us/step - loss: 0.1550 - val_loss: 0.1682
    Epoch 212/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 890us/step - loss: 0.1562 - val_loss: 0.1682
    Epoch 213/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 891us/step - loss: 0.1546 - val_loss: 0.1682
    Epoch 214/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 894us/step - loss: 0.1544 - val_loss: 0.1683
    Epoch 215/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.1548 - val_loss: 0.1681
    Epoch 216/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 930us/step - loss: 0.1551 - val_loss: 0.1682
    Epoch 217/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 891us/step - loss: 0.1556 - val_loss: 0.1683
    Epoch 218/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 912us/step - loss: 0.1557 - val_loss: 0.1684
    Epoch 219/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1558 - val_loss: 0.1684
    Epoch 220/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1541 - val_loss: 0.1683
    Epoch 221/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 891us/step - loss: 0.1552 - val_loss: 0.1683
    Epoch 222/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 895us/step - loss: 0.1557 - val_loss: 0.1682
    Epoch 223/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 911us/step - loss: 0.1546 - val_loss: 0.1683
    Epoch 224/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 925us/step - loss: 0.1549 - val_loss: 0.1683
    Epoch 225/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 884us/step - loss: 0.1544 - val_loss: 0.1686
    Epoch 226/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 903us/step - loss: 0.1539 - val_loss: 0.1685
    Epoch 227/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 914us/step - loss: 0.1541 - val_loss: 0.1684
    Epoch 228/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 902us/step - loss: 0.1545 - val_loss: 0.1684
    Epoch 229/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 918us/step - loss: 0.1542 - val_loss: 0.1684
    Epoch 230/1000
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 898us/step - loss: 0.1539 - val_loss: 0.1685


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

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 



    
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
    

