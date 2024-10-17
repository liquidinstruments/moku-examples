# Outputting a weighted sum of inputs

In this tutorial, you will learn the fundamentals of using the provided neural network library for training a Moku Neural Network to deploy on Moku:Pro. This library uses the [Tensorflow](https://www.tensorflow.org/) implementation of [Keras](https://www.tensorflow.org/guide/keras) to instill best practices for deploying an FPGA-based neural network. In this example, we will train a basic model to output a weighted sum of the input chnnels with an optional bias/offset term using a single layer with a single neuron.


```python
import numpy as np
import matplotlib.pyplot as plt

from moku.nn import LinnModel, save_linn
```

# Step 1: Generate input (3 channels) and output (one channel) data for training

Generate training data and plot the simulated signals to verify them. You can also choose to use Moku to produce training data.


```python
nPts    = 1024
nCycles = 5
x       = np.linspace(0, nCycles * 2 * np.pi, nPts)

in1 = np.sin(x)

in2 = np.sign(in1)

in3 = 1/np.pi * ((x % (2*np.pi)) - np.pi)

offset = 0.1
amp1   = 0.1
amp2   = 0.2
amp3   = 0.3

out = amp1*in1 + amp2*in2 + amp3*in3 + offset

plt.figure()
plt.plot(in1)
plt.plot(in2)
plt.plot(in3)
plt.title('Inputs')

plt.figure()
plt.plot(out)
plt.title('Desired output')
```




    Text(0.5, 1.0, 'Desired output')




    
![png](./Sum_files/Sum_3_1.png)
    



    
![png](./Sum_files/Sum_3_2.png)
    


# Step 2: Defining the model and train the neural network

Now that we have defined the training data we need to define a model which we will then subsequently train. A neural network is made of succesive layers of connected neurons that implement a generally non-linear mapping of a linear transform: $A(Wx + b)$, where $W$ is the weight matrix, $x$ is an input vector, $b$ is a bias vector and $A$ is the activation function. By successively stacking and connecting arbitrarily large numbers of artificial neurons, a neural network becomes a unviersal approximator. While there are many different types of neural networks, the Moku FPGA currently only supports densely connected feedforward networks. More information on these type of networks, otherwise known as multilayer perceptrons, can be found here: [Goodfellow-Ch6](https://www.deeplearningbook.org/contents/mlp.html).

We also reshape the data to reflect that we have N number of training examples of size 1. In general, training data should allow for the mapping $(N, M)\mapsto(N, K)$ where N is the number of training examples, M is the input feature dimension and K is the output feature dimension. In this case we are looking to map $X \mapsto Y$ where both X and Y have shape (N, 1).



```python
# Reshape the inputs, ready for training. Each `data point' is three inputs and one output
inputs = np.vstack((in1, in2, in3)).transpose() # Shape is now nPts x 3
out.shape = [nPts, 1] # Shape is now nPts x 1

```

We need to define the model structure that we will use to represent the functional mapping. To do this, we will define a simple model with only one dense layer of a single neuron and a linear activation function. We provide a model definition as a list of tuples, where each tuple can take the form: `(layer_size, activation)`.

Build the neural network model. Skip the I/O scaling as the data is sufficient for training,
Note that if you add scaling here, you'll need to apply the same scaling in the Moku Neural Network instrument at
runtime.


```python
linn_model = LinnModel()
linn_model.set_training_data(inputs, out, scale=False)

model_definition = [ (1, 'linear')] # A linear model should give a perfect prediction in this contrived case
                                    # Try some of the others below

# model_definition = [ (4, 'relu'), (4, 'relu')] # ReLU activation not great on signal reconstruction
# model_definition = [ (4, 'tanh'), (4, 'tanh')] # tanh has a range of +-1 which is more "voltage-like"
# model_definition = [ (16, 'tanh'), (16, 'tanh')] # A few more degrees of freedom to play around with
# model_definition = [ (100, 'tanh'), (100, 'tanh'), (100, 'tanh'), (100, 'tanh'), (100, 'tanh')] # Biggest Moku can fit, can overfit!

linn_model.construct_model(model_definition)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">4</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">4</span> (16.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4</span> (16.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Here we can see our dense layers that match the size that we defined. There are a number of intermediate layers listed as `output_clip_layer` which can be ignored. These layers are a byproduct of the quantisation for the Moku implmentation and are automatically added by the quantised model.

Now that we have defined our model, we need to train it so that it will represent our desired mapping. This is as simple as calling the `fit_model()` function with a few simple arguments. We will allow our model to train for 500 epochs (training steps) and will pass our validation data that we used earlier.

<div class="alert alert-block alert-info">
<b>Tip:</b> For those familiar with Keras, keyword arguments can be passed to the Keras 'fit' function via the 'fit_model' function.
</div>


```python
# %%
# Train the model. This simple model converges quickly, so an early stopping config terminates training much more quickly.
history = linn_model.fit_model(epochs=500, validation_split=0.1, es_config={'patience': 10})

```

    Value for monitor missing. Using default:val_loss.
    Value for restore missing. Using default:False.


    Epoch 1/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - loss: 0.6554 - val_loss: 0.4668
    Epoch 2/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 809us/step - loss: 0.5419 - val_loss: 0.4265
    Epoch 3/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 826us/step - loss: 0.4624 - val_loss: 0.3913
    Epoch 4/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.4268 - val_loss: 0.3574
    Epoch 5/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.3563 - val_loss: 0.3288
    Epoch 6/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.3183 - val_loss: 0.3044
    Epoch 7/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.2642 - val_loss: 0.2809
    Epoch 8/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.2572 - val_loss: 0.2592
    Epoch 9/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.2224 - val_loss: 0.2432
    Epoch 10/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 0.2183 - val_loss: 0.2269
    Epoch 11/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.1941 - val_loss: 0.2148
    Epoch 12/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 814us/step - loss: 0.1823 - val_loss: 0.2041
    Epoch 13/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.1773 - val_loss: 0.1938
    Epoch 14/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.1688 - val_loss: 0.1870
    Epoch 15/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.1546 - val_loss: 0.1792
    Epoch 16/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.1522 - val_loss: 0.1737
    Epoch 17/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.1461 - val_loss: 0.1677
    Epoch 18/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.1502 - val_loss: 0.1634
    Epoch 19/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.1408 - val_loss: 0.1586
    Epoch 20/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.1292 - val_loss: 0.1546
    Epoch 21/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.1255 - val_loss: 0.1504
    Epoch 22/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.1238 - val_loss: 0.1467
    Epoch 23/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 722us/step - loss: 0.1194 - val_loss: 0.1432
    Epoch 24/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.1299 - val_loss: 0.1395
    Epoch 25/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.1218 - val_loss: 0.1363
    Epoch 26/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.1146 - val_loss: 0.1330
    Epoch 27/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.1067 - val_loss: 0.1298
    Epoch 28/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 0.1112 - val_loss: 0.1268
    Epoch 29/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.1104 - val_loss: 0.1238
    Epoch 30/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0976 - val_loss: 0.1208
    Epoch 31/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.1014 - val_loss: 0.1179
    Epoch 32/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.1029 - val_loss: 0.1151
    Epoch 33/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0928 - val_loss: 0.1125
    Epoch 34/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 0.0931 - val_loss: 0.1094
    Epoch 35/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0882 - val_loss: 0.1068
    Epoch 36/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0844 - val_loss: 0.1045
    Epoch 37/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0829 - val_loss: 0.1020
    Epoch 38/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0804 - val_loss: 0.0992
    Epoch 39/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0811 - val_loss: 0.0970
    Epoch 40/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0720 - val_loss: 0.0945
    Epoch 41/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0640 - val_loss: 0.0925
    Epoch 42/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 0.0763 - val_loss: 0.0899
    Epoch 43/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.0743 - val_loss: 0.0878
    Epoch 44/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 843us/step - loss: 0.0674 - val_loss: 0.0857
    Epoch 45/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0626 - val_loss: 0.0836
    Epoch 46/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 810us/step - loss: 0.0643 - val_loss: 0.0816
    Epoch 47/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0552 - val_loss: 0.0794
    Epoch 48/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0628 - val_loss: 0.0768
    Epoch 49/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 815us/step - loss: 0.0569 - val_loss: 0.0743
    Epoch 50/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 0.0575 - val_loss: 0.0720
    Epoch 51/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 890us/step - loss: 0.0542 - val_loss: 0.0695
    Epoch 52/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 857us/step - loss: 0.0494 - val_loss: 0.0670
    Epoch 53/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 0.0522 - val_loss: 0.0649
    Epoch 54/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 814us/step - loss: 0.0477 - val_loss: 0.0627
    Epoch 55/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 819us/step - loss: 0.0450 - val_loss: 0.0606
    Epoch 56/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 862us/step - loss: 0.0470 - val_loss: 0.0585
    Epoch 57/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 827us/step - loss: 0.0434 - val_loss: 0.0566
    Epoch 58/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 821us/step - loss: 0.0417 - val_loss: 0.0546
    Epoch 59/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 961us/step - loss: 0.0424 - val_loss: 0.0527
    Epoch 60/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0394 - val_loss: 0.0509
    Epoch 61/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 0.0392 - val_loss: 0.0490
    Epoch 62/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 823us/step - loss: 0.0372 - val_loss: 0.0474
    Epoch 63/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 821us/step - loss: 0.0381 - val_loss: 0.0456
    Epoch 64/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.0343 - val_loss: 0.0439
    Epoch 65/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 0.0319 - val_loss: 0.0423
    Epoch 66/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.0314 - val_loss: 0.0409
    Epoch 67/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0320 - val_loss: 0.0391
    Epoch 68/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 835us/step - loss: 0.0296 - val_loss: 0.0378
    Epoch 69/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0266 - val_loss: 0.0365
    Epoch 70/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 807us/step - loss: 0.0282 - val_loss: 0.0350
    Epoch 71/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 798us/step - loss: 0.0271 - val_loss: 0.0337
    Epoch 72/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0255 - val_loss: 0.0324
    Epoch 73/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.0236 - val_loss: 0.0312
    Epoch 74/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 811us/step - loss: 0.0236 - val_loss: 0.0300
    Epoch 75/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 817us/step - loss: 0.0234 - val_loss: 0.0289
    Epoch 76/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 797us/step - loss: 0.0227 - val_loss: 0.0276
    Epoch 77/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0199 - val_loss: 0.0266
    Epoch 78/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 0.0194 - val_loss: 0.0254
    Epoch 79/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 0.0202 - val_loss: 0.0245
    Epoch 80/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0189 - val_loss: 0.0234
    Epoch 81/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0164 - val_loss: 0.0225
    Epoch 82/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.0161 - val_loss: 0.0216
    Epoch 83/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 934us/step - loss: 0.0161 - val_loss: 0.0206
    Epoch 84/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.0163 - val_loss: 0.0197
    Epoch 85/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0158 - val_loss: 0.0189
    Epoch 86/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0143 - val_loss: 0.0180
    Epoch 87/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.0142 - val_loss: 0.0172
    Epoch 88/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0131 - val_loss: 0.0165
    Epoch 89/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 808us/step - loss: 0.0134 - val_loss: 0.0158
    Epoch 90/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0121 - val_loss: 0.0151
    Epoch 91/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 0.0116 - val_loss: 0.0144
    Epoch 92/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0108 - val_loss: 0.0137
    Epoch 93/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 0.0103 - val_loss: 0.0130
    Epoch 94/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.0099 - val_loss: 0.0124
    Epoch 95/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0096 - val_loss: 0.0119
    Epoch 96/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0093 - val_loss: 0.0113
    Epoch 97/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0092 - val_loss: 0.0108
    Epoch 98/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 890us/step - loss: 0.0079 - val_loss: 0.0102
    Epoch 99/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 0.0072 - val_loss: 0.0097
    Epoch 100/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.0082 - val_loss: 0.0092
    Epoch 101/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.0075 - val_loss: 0.0087
    Epoch 102/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0067 - val_loss: 0.0083
    Epoch 103/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0058 - val_loss: 0.0079
    Epoch 104/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 0.0066 - val_loss: 0.0074
    Epoch 105/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0059 - val_loss: 0.0070
    Epoch 106/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0055 - val_loss: 0.0066
    Epoch 107/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 884us/step - loss: 0.0053 - val_loss: 0.0063
    Epoch 108/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 810us/step - loss: 0.0045 - val_loss: 0.0059
    Epoch 109/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 807us/step - loss: 0.0045 - val_loss: 0.0056
    Epoch 110/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 865us/step - loss: 0.0044 - val_loss: 0.0052
    Epoch 111/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 841us/step - loss: 0.0039 - val_loss: 0.0050
    Epoch 112/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 0.0039 - val_loss: 0.0047
    Epoch 113/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 0.0036 - val_loss: 0.0044
    Epoch 114/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0033 - val_loss: 0.0041
    Epoch 115/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0034 - val_loss: 0.0039
    Epoch 116/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 984us/step - loss: 0.0031 - val_loss: 0.0036
    Epoch 117/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 807us/step - loss: 0.0030 - val_loss: 0.0034
    Epoch 118/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 0.0026 - val_loss: 0.0032
    Epoch 119/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0026 - val_loss: 0.0030
    Epoch 120/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0023 - val_loss: 0.0028
    Epoch 121/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 0.0023 - val_loss: 0.0026
    Epoch 122/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 0.0022 - val_loss: 0.0024
    Epoch 123/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0019 - val_loss: 0.0023
    Epoch 124/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0018 - val_loss: 0.0021
    Epoch 125/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 0.0016 - val_loss: 0.0020
    Epoch 126/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.0015 - val_loss: 0.0018
    Epoch 127/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0015 - val_loss: 0.0017
    Epoch 128/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 0.0014 - val_loss: 0.0016
    Epoch 129/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0012 - val_loss: 0.0014
    Epoch 130/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 0.0011 - val_loss: 0.0013
    Epoch 131/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0011 - val_loss: 0.0012
    Epoch 132/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 808us/step - loss: 0.0011 - val_loss: 0.0011
    Epoch 133/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 912us/step - loss: 8.2786e-04 - val_loss: 0.0011
    Epoch 134/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 9.5697e-04 - val_loss: 9.6261e-04
    Epoch 135/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.6418e-04 - val_loss: 8.8554e-04
    Epoch 136/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 7.2838e-04 - val_loss: 8.2821e-04
    Epoch 137/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 6.7495e-04 - val_loss: 7.4096e-04
    Epoch 138/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 6.1193e-04 - val_loss: 6.9471e-04
    Epoch 139/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 5.1053e-04 - val_loss: 6.2748e-04
    Epoch 140/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 4.6720e-04 - val_loss: 5.7495e-04
    Epoch 141/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 4.3826e-04 - val_loss: 5.2681e-04
    Epoch 142/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 4.0303e-04 - val_loss: 4.7773e-04
    Epoch 143/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 3.6673e-04 - val_loss: 4.3709e-04
    Epoch 144/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 3.2945e-04 - val_loss: 3.9512e-04
    Epoch 145/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 3.0745e-04 - val_loss: 3.5861e-04
    Epoch 146/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 2.7197e-04 - val_loss: 3.2522e-04
    Epoch 147/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 2.5440e-04 - val_loss: 2.9519e-04
    Epoch 148/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 2.2594e-04 - val_loss: 2.6884e-04
    Epoch 149/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 1.9712e-04 - val_loss: 2.4185e-04
    Epoch 150/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 865us/step - loss: 1.8323e-04 - val_loss: 2.1547e-04
    Epoch 151/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 1.6071e-04 - val_loss: 1.9614e-04
    Epoch 152/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 1.5292e-04 - val_loss: 1.7551e-04
    Epoch 153/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 1.2909e-04 - val_loss: 1.5641e-04
    Epoch 154/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 732us/step - loss: 1.1908e-04 - val_loss: 1.4020e-04
    Epoch 155/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 709us/step - loss: 1.0988e-04 - val_loss: 1.2596e-04
    Epoch 156/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 8.8669e-05 - val_loss: 1.1305e-04
    Epoch 157/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 9.4077e-05 - val_loss: 1.0018e-04
    Epoch 158/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 7.8013e-05 - val_loss: 8.8653e-05
    Epoch 159/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 6.6745e-05 - val_loss: 7.8766e-05
    Epoch 160/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 5.9114e-05 - val_loss: 7.0359e-05
    Epoch 161/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 890us/step - loss: 5.3357e-05 - val_loss: 6.1653e-05
    Epoch 162/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 4.6371e-05 - val_loss: 5.4563e-05
    Epoch 163/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 4.2904e-05 - val_loss: 4.8170e-05
    Epoch 164/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 3.9236e-05 - val_loss: 4.2199e-05
    Epoch 165/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 3.4991e-05 - val_loss: 3.7200e-05
    Epoch 166/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 3.0668e-05 - val_loss: 3.2677e-05
    Epoch 167/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 880us/step - loss: 2.4791e-05 - val_loss: 2.8391e-05
    Epoch 168/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 2.0153e-05 - val_loss: 2.5137e-05
    Epoch 169/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 814us/step - loss: 2.0067e-05 - val_loss: 2.1677e-05
    Epoch 170/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 798us/step - loss: 1.5821e-05 - val_loss: 1.8912e-05
    Epoch 171/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 1.4645e-05 - val_loss: 1.6326e-05
    Epoch 172/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 1.2214e-05 - val_loss: 1.4129e-05
    Epoch 173/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 1.1334e-05 - val_loss: 1.2285e-05
    Epoch 174/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 8.5648e-06 - val_loss: 1.0507e-05
    Epoch 175/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 8.6203e-06 - val_loss: 9.0595e-06
    Epoch 176/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 6.6976e-06 - val_loss: 7.7965e-06
    Epoch 177/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 5.7772e-06 - val_loss: 6.6624e-06
    Epoch 178/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 5.3360e-06 - val_loss: 5.6124e-06
    Epoch 179/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 4.0281e-06 - val_loss: 4.8533e-06
    Epoch 180/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 3.7121e-06 - val_loss: 4.0278e-06
    Epoch 181/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 2.9814e-06 - val_loss: 3.4904e-06
    Epoch 182/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 2.8084e-06 - val_loss: 2.9019e-06
    Epoch 183/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 2.5154e-06 - val_loss: 2.4919e-06
    Epoch 184/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 1.9748e-06 - val_loss: 2.1190e-06
    Epoch 185/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 901us/step - loss: 1.5626e-06 - val_loss: 1.7543e-06
    Epoch 186/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 1.4886e-06 - val_loss: 1.4474e-06
    Epoch 187/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 1.1827e-06 - val_loss: 1.2282e-06
    Epoch 188/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 1.0115e-06 - val_loss: 1.0145e-06
    Epoch 189/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 7.5474e-07 - val_loss: 8.4916e-07
    Epoch 190/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 6.0835e-07 - val_loss: 6.9621e-07
    Epoch 191/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 5.4741e-07 - val_loss: 5.7570e-07
    Epoch 192/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 4.2231e-07 - val_loss: 4.7287e-07
    Epoch 193/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 3.9057e-07 - val_loss: 3.8841e-07
    Epoch 194/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 776us/step - loss: 2.8721e-07 - val_loss: 3.1895e-07
    Epoch 195/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 2.3331e-07 - val_loss: 2.5894e-07
    Epoch 196/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 797us/step - loss: 1.9116e-07 - val_loss: 2.1203e-07
    Epoch 197/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 1.7241e-07 - val_loss: 1.6940e-07
    Epoch 198/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 860us/step - loss: 1.3342e-07 - val_loss: 1.3888e-07
    Epoch 199/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 1.1388e-07 - val_loss: 1.0948e-07
    Epoch 200/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 8.3811e-08 - val_loss: 8.9784e-08
    Epoch 201/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 6.8186e-08 - val_loss: 7.0842e-08
    Epoch 202/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 5.1907e-08 - val_loss: 5.7970e-08
    Epoch 203/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 4.3853e-08 - val_loss: 4.5923e-08
    Epoch 204/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 898us/step - loss: 3.4191e-08 - val_loss: 3.5435e-08
    Epoch 205/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 2.5503e-08 - val_loss: 2.7988e-08
    Epoch 206/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 2.2247e-08 - val_loss: 2.1956e-08
    Epoch 207/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 1.6549e-08 - val_loss: 1.7288e-08
    Epoch 208/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 1.2833e-08 - val_loss: 1.3704e-08
    Epoch 209/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step - loss: 1.0866e-08 - val_loss: 1.0497e-08
    Epoch 210/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 8.4048e-09 - val_loss: 8.1005e-09
    Epoch 211/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 6.5375e-09 - val_loss: 6.2210e-09
    Epoch 212/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 4.5303e-09 - val_loss: 4.7762e-09
    Epoch 213/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 3.5089e-09 - val_loss: 3.6977e-09
    Epoch 214/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 3.0477e-09 - val_loss: 2.8032e-09
    Epoch 215/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 2.3931e-09 - val_loss: 2.1211e-09
    Epoch 216/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 1.5830e-09 - val_loss: 1.6016e-09
    Epoch 217/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 718us/step - loss: 1.2925e-09 - val_loss: 1.2094e-09
    Epoch 218/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 8.7733e-10 - val_loss: 9.0449e-10
    Epoch 219/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 6.7871e-10 - val_loss: 6.7260e-10
    Epoch 220/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 4.6978e-10 - val_loss: 5.0879e-10
    Epoch 221/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 910us/step - loss: 3.6597e-10 - val_loss: 3.6320e-10
    Epoch 222/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 2.6985e-10 - val_loss: 2.7417e-10
    Epoch 223/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 1.8850e-10 - val_loss: 1.9754e-10
    Epoch 224/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 1.4897e-10 - val_loss: 1.4926e-10
    Epoch 225/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 1.0974e-10 - val_loss: 1.0307e-10
    Epoch 226/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 7.7623e-11 - val_loss: 7.6247e-11
    Epoch 227/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 5.7852e-11 - val_loss: 5.4413e-11
    Epoch 228/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 3.9390e-11 - val_loss: 3.8188e-11
    Epoch 229/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 2.6282e-11 - val_loss: 2.7788e-11
    Epoch 230/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 817us/step - loss: 2.0698e-11 - val_loss: 1.9101e-11
    Epoch 231/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 1.3556e-11 - val_loss: 1.3359e-11
    Epoch 232/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 1.1150e-11 - val_loss: 9.2519e-12
    Epoch 233/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.7752e-12 - val_loss: 6.4873e-12
    Epoch 234/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 802us/step - loss: 5.1644e-12 - val_loss: 4.5668e-12
    Epoch 235/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 808us/step - loss: 3.1617e-12 - val_loss: 3.0563e-12
    Epoch 236/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 2.3420e-12 - val_loss: 2.0482e-12
    Epoch 237/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 1.4662e-12 - val_loss: 1.4047e-12
    Epoch 238/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 9.3818e-13 - val_loss: 1.0573e-12
    Epoch 239/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 928us/step - loss: 7.6245e-13 - val_loss: 7.3083e-13
    Epoch 240/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 5.0674e-13 - val_loss: 4.8739e-13
    Epoch 241/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 3.3888e-13 - val_loss: 3.0649e-13
    Epoch 242/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 2.2161e-13 - val_loss: 2.2305e-13
    Epoch 243/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 1.6892e-13 - val_loss: 1.8584e-13
    Epoch 244/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 728us/step - loss: 1.4259e-13 - val_loss: 1.6312e-13
    Epoch 245/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 1.2204e-13 - val_loss: 1.4532e-13
    Epoch 246/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 1.2300e-13 - val_loss: 1.3006e-13
    Epoch 247/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 1.0147e-13 - val_loss: 1.3679e-13
    Epoch 248/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 1.0360e-13 - val_loss: 1.2970e-13
    Epoch 249/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 9.9198e-14 - val_loss: 1.1756e-13
    Epoch 250/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 8.3258e-14 - val_loss: 1.1212e-13
    Epoch 251/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 8.6305e-14 - val_loss: 1.0920e-13
    Epoch 252/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 8.6738e-14 - val_loss: 1.1212e-13
    Epoch 253/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 787us/step - loss: 8.2783e-14 - val_loss: 1.0740e-13
    Epoch 254/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 8.5224e-14 - val_loss: 8.8695e-14
    Epoch 255/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 6.3319e-14 - val_loss: 8.5224e-14
    Epoch 256/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 885us/step - loss: 6.3634e-14 - val_loss: 8.5224e-14
    Epoch 257/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 6.3500e-14 - val_loss: 8.5224e-14
    Epoch 258/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 6.2266e-14 - val_loss: 8.7071e-14
    Epoch 259/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 6.2595e-14 - val_loss: 8.4476e-14
    Epoch 260/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 785us/step - loss: 6.1930e-14 - val_loss: 8.2491e-14
    Epoch 261/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 6.5130e-14 - val_loss: 8.1957e-14
    Epoch 262/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 769us/step - loss: 7.1126e-14 - val_loss: 7.7124e-14
    Epoch 263/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 5.9198e-14 - val_loss: 7.6857e-14
    Epoch 264/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 868us/step - loss: 6.2444e-14 - val_loss: 7.7993e-14
    Epoch 265/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 5.6490e-14 - val_loss: 7.0027e-14
    Epoch 266/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 5.5555e-14 - val_loss: 6.5722e-14
    Epoch 267/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 817us/step - loss: 5.2430e-14 - val_loss: 6.5722e-14
    Epoch 268/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 5.2709e-14 - val_loss: 6.5722e-14
    Epoch 269/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 723us/step - loss: 4.9450e-14 - val_loss: 6.2103e-14
    Epoch 270/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 4.6971e-14 - val_loss: 6.2103e-14
    Epoch 271/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 800us/step - loss: 4.8091e-14 - val_loss: 6.5061e-14
    Epoch 272/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 4.4907e-14 - val_loss: 6.2720e-14
    Epoch 273/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 4.9583e-14 - val_loss: 5.1835e-14
    Epoch 274/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 941us/step - loss: 3.6729e-14 - val_loss: 4.1149e-14
    Epoch 275/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 3.2505e-14 - val_loss: 4.0980e-14
    Epoch 276/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 2.9616e-14 - val_loss: 3.9798e-14
    Epoch 277/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 3.0558e-14 - val_loss: 4.1724e-14
    Epoch 278/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 2.9737e-14 - val_loss: 3.9286e-14
    Epoch 279/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 3.0356e-14 - val_loss: 3.9286e-14
    Epoch 280/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 3.1849e-14 - val_loss: 3.7868e-14
    Epoch 281/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 3.1114e-14 - val_loss: 3.8898e-14
    Epoch 282/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 3.0753e-14 - val_loss: 4.2553e-14
    Epoch 283/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 3.1828e-14 - val_loss: 3.8832e-14
    Epoch 284/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 782us/step - loss: 2.9808e-14 - val_loss: 3.6991e-14
    Epoch 285/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 2.7718e-14 - val_loss: 3.6991e-14
    Epoch 286/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 2.7942e-14 - val_loss: 3.6991e-14
    Epoch 287/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 2.8685e-14 - val_loss: 3.8126e-14
    Epoch 288/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 2.9382e-14 - val_loss: 3.3023e-14
    Epoch 289/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 814us/step - loss: 1.9213e-14 - val_loss: 2.4907e-14
    Epoch 290/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 1.9303e-14 - val_loss: 2.6100e-14
    Epoch 291/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 840us/step - loss: 1.9531e-14 - val_loss: 2.5199e-14
    Epoch 292/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 2.1164e-14 - val_loss: 2.5199e-14
    Epoch 293/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 908us/step - loss: 1.9337e-14 - val_loss: 2.5199e-14
    Epoch 294/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 2.0866e-14 - val_loss: 2.6659e-14
    Epoch 295/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 2.1012e-14 - val_loss: 2.7216e-14
    Epoch 296/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 2.1105e-14 - val_loss: 2.5927e-14
    Epoch 297/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 1.9479e-14 - val_loss: 2.4970e-14
    Epoch 298/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 731us/step - loss: 1.8793e-14 - val_loss: 2.4406e-14
    Epoch 299/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 767us/step - loss: 1.9311e-14 - val_loss: 2.4622e-14
    Epoch 300/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 1.9570e-14 - val_loss: 2.4970e-14
    Epoch 301/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 2.0335e-14 - val_loss: 2.4970e-14
    Epoch 302/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 726us/step - loss: 1.8348e-14 - val_loss: 2.4622e-14
    Epoch 303/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 2.0352e-14 - val_loss: 1.8612e-14
    Epoch 304/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 1.4186e-14 - val_loss: 1.8612e-14
    Epoch 305/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 1.4020e-14 - val_loss: 1.8612e-14
    Epoch 306/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 1.4035e-14 - val_loss: 1.9746e-14
    Epoch 307/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 1.6141e-14 - val_loss: 1.8977e-14
    Epoch 308/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 1.4895e-14 - val_loss: 1.8030e-14
    Epoch 309/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 1.5124e-14 - val_loss: 1.8612e-14
    Epoch 310/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 791us/step - loss: 1.4875e-14 - val_loss: 2.0266e-14
    Epoch 311/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 1.5119e-14 - val_loss: 2.1340e-14
    Epoch 312/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 792us/step - loss: 1.5587e-14 - val_loss: 1.8365e-14
    Epoch 313/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 1.3827e-14 - val_loss: 1.2789e-14
    Epoch 314/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 1.0452e-14 - val_loss: 1.2621e-14
    Epoch 315/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 1.0569e-14 - val_loss: 1.3292e-14
    Epoch 316/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 1.0003e-14 - val_loss: 1.2053e-14
    Epoch 317/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 865us/step - loss: 1.0025e-14 - val_loss: 1.2621e-14
    Epoch 318/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 9.7318e-15 - val_loss: 9.5419e-15
    Epoch 319/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 7.2765e-15 - val_loss: 8.9054e-15
    Epoch 320/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 810us/step - loss: 7.4846e-15 - val_loss: 9.0807e-15
    Epoch 321/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 7.7584e-15 - val_loss: 9.0807e-15
    Epoch 322/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 6.4574e-15 - val_loss: 9.0807e-15
    Epoch 323/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 796us/step - loss: 6.9335e-15 - val_loss: 9.0807e-15
    Epoch 324/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 779us/step - loss: 7.4744e-15 - val_loss: 9.0807e-15
    Epoch 325/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 7.0852e-15 - val_loss: 1.1149e-14
    Epoch 326/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 8.1053e-15 - val_loss: 8.6259e-15
    Epoch 327/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 915us/step - loss: 7.5175e-15 - val_loss: 9.0807e-15
    Epoch 328/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 7.1673e-15 - val_loss: 9.0807e-15
    Epoch 329/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 833us/step - loss: 7.4133e-15 - val_loss: 8.9054e-15
    Epoch 330/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 780us/step - loss: 7.1016e-15 - val_loss: 8.9054e-15
    Epoch 331/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 6.7522e-15 - val_loss: 9.0807e-15
    Epoch 332/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 781us/step - loss: 7.0895e-15 - val_loss: 9.6419e-15
    Epoch 333/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 903us/step - loss: 7.3507e-15 - val_loss: 9.0807e-15
    Epoch 334/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 816us/step - loss: 6.7419e-15 - val_loss: 9.0807e-15
    Epoch 335/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 841us/step - loss: 7.2356e-15 - val_loss: 8.9054e-15
    Epoch 336/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 854us/step - loss: 6.8914e-15 - val_loss: 7.7649e-15
    Epoch 337/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 5.3285e-15 - val_loss: 3.6449e-15
    Epoch 338/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 836us/step - loss: 3.0766e-15 - val_loss: 3.8252e-15
    Epoch 339/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 842us/step - loss: 2.9277e-15 - val_loss: 3.8252e-15
    Epoch 340/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 799us/step - loss: 2.9385e-15 - val_loss: 3.8252e-15
    Epoch 341/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 3.0898e-15 - val_loss: 3.8252e-15
    Epoch 342/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 3.0271e-15 - val_loss: 3.8252e-15
    Epoch 343/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 885us/step - loss: 3.1485e-15 - val_loss: 3.8252e-15
    Epoch 344/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 818us/step - loss: 3.1089e-15 - val_loss: 3.6779e-15
    Epoch 345/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 815us/step - loss: 3.1501e-15 - val_loss: 3.6449e-15
    Epoch 346/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 838us/step - loss: 3.1991e-15 - val_loss: 3.8252e-15
    Epoch 347/500
    [1m29/29[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 829us/step - loss: 2.7716e-15 - val_loss: 3.8252e-15


We can use the returned history object to view the training and validation loss as a function of the training epochs. As training continues we can see that again the validation loss reduces and we end up with a model that has generalized to the training data.


```python
# plot the losses
plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.title('Loss functions')
plt.show()
```


    
![png](./Sum_files/Sum_11_0.png)
    


Optionally, we can view the performance of our model on our data by plotting the models predictions. In general, training an accurate model will be a function of many variables including choice of activation function, number of layers, number of neurons and the structure of the training data.


```python
# %%
# Plot the error between the actual and predicted points
nn_out = linn_model.predict(inputs)
fig, axs = plt.subplots(2)
fig.suptitle('Predicted output')
axs[0].plot(out,label='Desired')
axs[0].plot(nn_out,'--',label='Model output')
axs[0].legend()
axs[1].plot(nn_out-out,label='Model - Desired')
axs[1].legend()
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

```

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 330us/step



    
![png](./Sum_files/Sum_13_1.png)
    


Note the scale on the plot mapping the difference between model output and desired output.

# Step 3: Save the model to disk for use in the Moku Neural Network instrument.

The neural network is completely described by its weights and biases. In this simple case, we can inspect these values and compare them to the function that was used to generate the data in the first place. Note that it has almost exactly learned the original function.


```python
#Print model weights for interest and education
ii = 0
for layer in history.model.layers:
    if layer.get_weights():
        print(f'{ii}: ', 'Weights',layer.get_weights()[0].flatten().tolist(), ', Biases',layer.get_weights()[1].flatten().tolist())
        ii = ii + 1

# Save the model to a .linn file
save_linn(linn_model.model, input_channels=3, output_channels=1, file_name='Sum.linn')

```

    0:  Weights [0.1000000610947609, 0.1999998688697815, 0.29999983310699463] , Biases [0.10000000149011612]


    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 6 cycles



```python

```
