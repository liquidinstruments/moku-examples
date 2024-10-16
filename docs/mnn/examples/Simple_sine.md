# Introduction to moku-nn

In this basic tutorial we will go over the fundamentals of using the provided neural network library for training a neural network which can be deployed on the Moku. This library uses the [Tensorflow](https://www.tensorflow.org/) implementation of [Keras](https://www.tensorflow.org/guide/keras) to instill best practices for deploying an FPGA based neural network on the Moku. In this example, we will train a basic model to output a sine wave as a function of the input voltage on a given channel. This could be used to train a custom output response of the model as a function of input voltage (i.e. a complicated function that has no easy closed form expression), but here it will simply serve as an illustrative example.

<div class="alert alert-block alert-info">
<b>Tip:</b> For more experienced users, it is completely possible to use your favourite library to create your models for export to the Moku. In this case, however, it will be necessary to implement all of the relevant layer and output clipping, data scaling and layer constraints yourself.
</div>


```python
# import the relevant libraries for this example
import numpy as np
import matplotlib.pyplot as plt

from moku.nn import LinnModel, save_linn
# set the seed for repeatability
np.random.seed(7)
```

# Data Generation

The first step is to generate a data et that we will use to train the model. In this case we will generate an array of points from $0$ to $2\pi$ which will allow us to define "frequency" conveniently. In this case, we want our outputs to be a sin wave of frequency=5, such that if we were to sweep the voltage on the input of the Moku, we should output a sine wave with 5 cycles.


```python
# generate some very basic training data
X = np.arange(0, np.pi*2, 0.01)
Y = np.cos(X*5)

plt.plot(X, Y, '.')
plt.xticks(np.linspace(0, np.pi*2, 11), labels=["%.1f" % i for i in np.linspace(-1, 1, 11)])
plt.xlabel('Scaled input voltage (arb.)')
plt.ylabel('Scaled output voltage (arb.)')
plt.show()
```


    
![png](./Simple_sine_files/Simple_sine_5_0.png)
    


For best practice we wish to use the full range of the Moku's inputs to avoid quantisation errors. During export of the model, it will be assumed that the range of the inputs and outputs is `[-1, 1]`, with anything outside of this range being clipped. This assumption ensures at export that the full bit depth of the Moku is used to try and avoid quantisation errors. In this example, this assumes that the input voltage on the x-axis is being swept between the positive and negative limits of the Moku's input voltage range. Similarly, the output voltage is also assumed to be outputting from the negative and positive limits of the Moku's output voltage range. Maximum utilisation of the Moku bit depth will ensure that quantisation errors are minimised when exporting the floating point models to fixed point arrays required by the FPGA. As we will see in the next section, the quantised model implementation of the moku-nn library takes care of this process automatically, allowing you to work in the native data space.

To train the model correctly we will generate a training and validation set of data. The validation data is not used during the training process and allows us to measure the performance of the model without falling victim to overfitting. We will reserve 10% of the data for validaiton and randomly select samples from the training set to construct our validation set.

<div class="alert alert-block alert-info">
<b>Note:</b> You can skip this step and just pass the `validation_split` parameter to the `fit_model` function. Doing this explicitly allows us to plot the validation data below for interest and education.
</div>


```python
# get 10% of the random indices 
data_indices = np.arange(0, len(X), 1)
np.random.shuffle(data_indices)
val_length = int(len(X)*0.1) 
train_indices = data_indices[val_length:]
val_indices = data_indices[:val_length]

# separate the training and validation sets
train_X = X[train_indices]
train_Y = Y[train_indices]
val_X = X[val_indices]
val_Y = Y[val_indices]
```

# Defining the model

Now that we have defined the training data we need to define a model which we will then subsequently train. A neural network is made of succesive layers of connected neurons that implement a generally non-linear mapping of a linear transform: $A(Wx + b)$, where $W$ is the weight matrix, $x$ is an input vector, $b$ is a bias vector and $A$ is the activation function. By successively stacking and connecting arbitrarily large numbers of artificial neurons, a neural network becomes a unviersal approximator. While there are many different types of neural networks, the Moku FPGA currently only supports densely connected feedforward networks. More information on these type of networks, otherwise known as multilayer perceptrons, can be found here: [Goodfellow-Ch6](https://www.deeplearningbook.org/contents/mlp.html).

We start by instantiating the quantised model instance and passing the training data we created previously:


```python
# create the quantised model object
quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=train_X.reshape(-1,1), training_outputs=train_Y.reshape(-1,1))
```

Note that we are using unscaled values for the training data, instead this will be taken care of by the model object. Additionally we reshape the data to reflect that we have N number of training examples of size 1. In general, training data should allow for the mapping $(N, M)\mapsto(N, K)$ where N is the number of training examples, M is the input feature dimension and K is the output feature dimension. In this case we are looking to map $X \mapsto Y$ where both X and Y have shape (N, 1).

We need to define the model structure that we will use to represent the functional mapping. To do this we will define a model with 3 intermediate layers of size 32, where each layer uses the activation function ReLU. We provide a model definition as a list of tuples, where each tuple take the form: `(layer_size, activation)`. We will use the `show_summary` flag to map to output the constructed model summary.


```python
# model definition
model_definition = [(32, 'relu'), (32, 'relu'), (32, 'tanh'), (1, 'linear')]

# build the model
quant_mod.construct_model(model_definition, show_summary=True)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_12"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_51 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">64</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_51            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_52 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_52            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_53 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_53            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_54 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_54            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,209</span> (8.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,209</span> (8.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



Here we can see our dense layers that match the size that we defined. There are a number of intermediate layers listed as `output_clip_layer` which can be ignored. These layers are a byproduct of the quantisation for the Moku implmentation and are automatically added by the quantised model.

# Training the model

Now that we have defined our model, we need to train it so that it will represent our desired mapping. This is as simple as calling the `fit_model()` function with a few simple arguments. We will allow our model to train for 600 epochs (training steps) and will pass our validation data that we used earlier.

<div class="alert alert-block alert-info">
<b>Tip:</b> For those familiar with Keras, keyword arguments can be passed to the Keras 'fit' function via the 'fit_model' function.
</div>


```python
# fit the model
history = quant_mod.fit_model(epochs=1000, validation_data=(val_X.reshape(-1,1), val_Y.reshape(-1,1)))
```

    Epoch 1/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - loss: 0.5099 - val_loss: 0.4629
    Epoch 2/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.5233 - val_loss: 0.4943
    Epoch 3/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4758 - val_loss: 0.4779
    Epoch 4/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.5120 - val_loss: 0.4759
    Epoch 5/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4855 - val_loss: 0.4635
    Epoch 6/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4719 - val_loss: 0.5107
    Epoch 7/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4754 - val_loss: 0.4832
    Epoch 8/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4632 - val_loss: 0.4964
    Epoch 9/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4765 - val_loss: 0.4914
    Epoch 10/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4674 - val_loss: 0.4874
    Epoch 11/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4433 - val_loss: 0.5148
    Epoch 12/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4535 - val_loss: 0.5525
    Epoch 13/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4312 - val_loss: 0.5041
    Epoch 14/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4471 - val_loss: 0.5784
    Epoch 15/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.4372 - val_loss: 0.5588
    Epoch 16/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4234 - val_loss: 0.6038
    Epoch 17/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4486 - val_loss: 0.5890
    Epoch 18/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4413 - val_loss: 0.5896
    Epoch 19/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4129 - val_loss: 0.5767
    Epoch 20/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4355 - val_loss: 0.6675
    Epoch 21/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4419 - val_loss: 0.6300
    Epoch 22/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4311 - val_loss: 0.5455
    Epoch 23/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4287 - val_loss: 0.6209
    Epoch 24/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4139 - val_loss: 0.5499
    Epoch 25/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4369 - val_loss: 0.6390
    Epoch 26/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4282 - val_loss: 0.5225
    Epoch 27/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4326 - val_loss: 0.5860
    Epoch 28/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4364 - val_loss: 0.5167
    Epoch 29/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4352 - val_loss: 0.5268
    Epoch 30/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4362 - val_loss: 0.5220
    Epoch 31/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4291 - val_loss: 0.5418
    Epoch 32/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4291 - val_loss: 0.5199
    Epoch 33/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4000 - val_loss: 0.5421
    Epoch 34/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4070 - val_loss: 0.5165
    Epoch 35/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4204 - val_loss: 0.5095
    Epoch 36/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4092 - val_loss: 0.5491
    Epoch 37/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4105 - val_loss: 0.5135
    Epoch 38/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4073 - val_loss: 0.6493
    Epoch 39/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.4231 - val_loss: 0.5323
    Epoch 40/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3884 - val_loss: 0.7370
    Epoch 41/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4014 - val_loss: 0.6445
    Epoch 42/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3904 - val_loss: 1.2787
    Epoch 43/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3665 - val_loss: 1.0065
    Epoch 44/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3716 - val_loss: 0.7737
    Epoch 45/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4050 - val_loss: 1.4253
    Epoch 46/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3829 - val_loss: 1.0748
    Epoch 47/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3702 - val_loss: 0.9019
    Epoch 48/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3627 - val_loss: 1.3118
    Epoch 49/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3744 - val_loss: 1.3032
    Epoch 50/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3645 - val_loss: 1.3407
    Epoch 51/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3728 - val_loss: 1.3122
    Epoch 52/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3473 - val_loss: 1.3531
    Epoch 53/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3457 - val_loss: 1.3740
    Epoch 54/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3459 - val_loss: 1.4981
    Epoch 55/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3539 - val_loss: 1.3926
    Epoch 56/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.3523 - val_loss: 1.3959
    Epoch 57/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3235 - val_loss: 1.5007
    Epoch 58/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3475 - val_loss: 1.5077
    Epoch 59/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3459 - val_loss: 1.2817
    Epoch 60/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3697 - val_loss: 1.3879
    Epoch 61/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3500 - val_loss: 1.3513
    Epoch 62/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3590 - val_loss: 1.3805
    Epoch 63/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3500 - val_loss: 1.3793
    Epoch 64/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3807 - val_loss: 1.3363
    Epoch 65/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3488 - val_loss: 1.3948
    Epoch 66/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3196 - val_loss: 1.4215
    Epoch 67/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3273 - val_loss: 1.4485
    Epoch 68/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3267 - val_loss: 1.4650
    Epoch 69/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3463 - val_loss: 1.3691
    Epoch 70/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3346 - val_loss: 1.4596
    Epoch 71/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3097 - val_loss: 1.4424
    Epoch 72/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3458 - val_loss: 1.4316
    Epoch 73/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3427 - val_loss: 1.4831
    Epoch 74/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3324 - val_loss: 1.4798
    Epoch 75/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3150 - val_loss: 1.4580
    Epoch 76/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3399 - val_loss: 1.4691
    Epoch 77/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3203 - val_loss: 1.4292
    Epoch 78/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3216 - val_loss: 1.4548
    Epoch 79/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3287 - val_loss: 1.4463
    Epoch 80/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3253 - val_loss: 1.4643
    Epoch 81/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3438 - val_loss: 1.4866
    Epoch 82/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3435 - val_loss: 1.4949
    Epoch 83/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3383 - val_loss: 1.4915
    Epoch 84/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3302 - val_loss: 1.4731
    Epoch 85/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3376 - val_loss: 1.4723
    Epoch 86/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3387 - val_loss: 1.4534
    Epoch 87/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3263 - val_loss: 1.4590
    Epoch 88/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3281 - val_loss: 1.4669
    Epoch 89/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3384 - val_loss: 1.4975
    Epoch 90/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3419 - val_loss: 1.4312
    Epoch 91/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3198 - val_loss: 1.4911
    Epoch 92/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3370 - val_loss: 1.4506
    Epoch 93/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3651 - val_loss: 1.4808
    Epoch 94/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3142 - val_loss: 1.4193
    Epoch 95/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3141 - val_loss: 1.4034
    Epoch 96/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3154 - val_loss: 1.4924
    Epoch 97/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3066 - val_loss: 1.4025
    Epoch 98/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.3144 - val_loss: 1.4883
    Epoch 99/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3343 - val_loss: 1.4455
    Epoch 100/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3423 - val_loss: 1.4578
    Epoch 101/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3539 - val_loss: 1.4024
    Epoch 102/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3307 - val_loss: 1.4800
    Epoch 103/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3405 - val_loss: 1.4764
    Epoch 104/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2946 - val_loss: 1.3924
    Epoch 105/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3281 - val_loss: 1.4339
    Epoch 106/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3587 - val_loss: 1.4953
    Epoch 107/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3421 - val_loss: 1.4461
    Epoch 108/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3204 - val_loss: 1.4819
    Epoch 109/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3130 - val_loss: 1.3804
    Epoch 110/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3088 - val_loss: 1.4643
    Epoch 111/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3285 - val_loss: 1.4772
    Epoch 112/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3222 - val_loss: 1.4328
    Epoch 113/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3405 - val_loss: 1.4197
    Epoch 114/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.3312 - val_loss: 1.3808
    Epoch 115/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3271 - val_loss: 1.4818
    Epoch 116/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3683 - val_loss: 1.4749
    Epoch 117/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3289 - val_loss: 1.4331
    Epoch 118/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3309 - val_loss: 1.4942
    Epoch 119/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.3258 - val_loss: 1.3945
    Epoch 120/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3315 - val_loss: 1.4726
    Epoch 121/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3326 - val_loss: 1.4167
    Epoch 122/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3075 - val_loss: 1.4088
    Epoch 123/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3076 - val_loss: 1.4134
    Epoch 124/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3423 - val_loss: 0.9973
    Epoch 125/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3410 - val_loss: 1.4260
    Epoch 126/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3039 - val_loss: 1.4631
    Epoch 127/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3200 - val_loss: 1.4364
    Epoch 128/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3046 - val_loss: 1.0378
    Epoch 129/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3157 - val_loss: 1.4690
    Epoch 130/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3069 - val_loss: 1.1602
    Epoch 131/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.3181 - val_loss: 1.0652
    Epoch 132/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3085 - val_loss: 0.9444
    Epoch 133/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3030 - val_loss: 0.9764
    Epoch 134/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2959 - val_loss: 0.6063
    Epoch 135/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2740 - val_loss: 0.5251
    Epoch 136/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2894 - val_loss: 0.5499
    Epoch 137/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2670 - val_loss: 0.6627
    Epoch 138/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3026 - val_loss: 0.6603
    Epoch 139/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2582 - val_loss: 0.6137
    Epoch 140/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2638 - val_loss: 0.7224
    Epoch 141/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2719 - val_loss: 0.6738
    Epoch 142/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2698 - val_loss: 0.7472
    Epoch 143/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3071 - val_loss: 0.8601
    Epoch 144/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2375 - val_loss: 0.9208
    Epoch 145/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2682 - val_loss: 0.9074
    Epoch 146/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2558 - val_loss: 0.9397
    Epoch 147/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2466 - val_loss: 0.9721
    Epoch 148/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2500 - val_loss: 0.9370
    Epoch 149/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2510 - val_loss: 0.9677
    Epoch 150/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2623 - val_loss: 1.0879
    Epoch 151/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2362 - val_loss: 1.0906
    Epoch 152/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2679 - val_loss: 1.0292
    Epoch 153/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2406 - val_loss: 1.0880
    Epoch 154/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.3005 - val_loss: 1.0864
    Epoch 155/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2115 - val_loss: 1.0889
    Epoch 156/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2643 - val_loss: 1.0807
    Epoch 157/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2354 - val_loss: 1.0402
    Epoch 158/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2548 - val_loss: 1.0562
    Epoch 159/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2769 - val_loss: 1.0221
    Epoch 160/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2376 - val_loss: 1.0985
    Epoch 161/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2387 - val_loss: 1.1162
    Epoch 162/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2545 - val_loss: 1.0619
    Epoch 163/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2470 - val_loss: 1.1031
    Epoch 164/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2605 - val_loss: 1.0973
    Epoch 165/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2551 - val_loss: 1.0991
    Epoch 166/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2540 - val_loss: 1.0612
    Epoch 167/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2501 - val_loss: 1.1282
    Epoch 168/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2455 - val_loss: 1.1068
    Epoch 169/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2209 - val_loss: 1.1012
    Epoch 170/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2316 - val_loss: 1.1066
    Epoch 171/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2321 - val_loss: 1.1143
    Epoch 172/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2459 - val_loss: 1.1123
    Epoch 173/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2364 - val_loss: 1.0915
    Epoch 174/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2248 - val_loss: 1.1373
    Epoch 175/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2285 - val_loss: 1.1039
    Epoch 176/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2429 - val_loss: 1.1121
    Epoch 177/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2301 - val_loss: 1.1190
    Epoch 178/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2416 - val_loss: 1.1176
    Epoch 179/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2187 - val_loss: 1.1166
    Epoch 180/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2662 - val_loss: 1.1168
    Epoch 181/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2525 - val_loss: 1.1197
    Epoch 182/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2385 - val_loss: 1.1149
    Epoch 183/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.2221 - val_loss: 1.1295
    Epoch 184/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2324 - val_loss: 1.1324
    Epoch 185/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2785 - val_loss: 1.1055
    Epoch 186/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2543 - val_loss: 1.1126
    Epoch 187/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2581 - val_loss: 1.1190
    Epoch 188/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2067 - val_loss: 1.1125
    Epoch 189/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2388 - val_loss: 1.1230
    Epoch 190/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2108 - val_loss: 1.1177
    Epoch 191/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2469 - val_loss: 1.1271
    Epoch 192/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2242 - val_loss: 1.1108
    Epoch 193/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2621 - val_loss: 1.1195
    Epoch 194/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2534 - val_loss: 1.1214
    Epoch 195/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2341 - val_loss: 1.1130
    Epoch 196/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2291 - val_loss: 1.1283
    Epoch 197/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2466 - val_loss: 1.1143
    Epoch 198/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2498 - val_loss: 1.1287
    Epoch 199/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2143 - val_loss: 1.1200
    Epoch 200/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2077 - val_loss: 1.1214
    Epoch 201/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2265 - val_loss: 1.1218
    Epoch 202/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2588 - val_loss: 1.1222
    Epoch 203/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2270 - val_loss: 1.0922
    Epoch 204/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2022 - val_loss: 1.1411
    Epoch 205/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2297 - val_loss: 1.1113
    Epoch 206/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2376 - val_loss: 1.1098
    Epoch 207/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2486 - val_loss: 1.0863
    Epoch 208/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2245 - val_loss: 1.0729
    Epoch 209/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2122 - val_loss: 1.0371
    Epoch 210/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2334 - val_loss: 1.0642
    Epoch 211/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2183 - val_loss: 1.0643
    Epoch 212/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2272 - val_loss: 0.9705
    Epoch 213/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2410 - val_loss: 0.9438
    Epoch 214/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2382 - val_loss: 0.9327
    Epoch 215/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2189 - val_loss: 0.8756
    Epoch 216/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2415 - val_loss: 0.8165
    Epoch 217/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2411 - val_loss: 0.8716
    Epoch 218/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2309 - val_loss: 0.7110
    Epoch 219/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2475 - val_loss: 0.7016
    Epoch 220/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2344 - val_loss: 0.7794
    Epoch 221/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2177 - val_loss: 0.6328
    Epoch 222/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2188 - val_loss: 0.6498
    Epoch 223/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2231 - val_loss: 0.6487
    Epoch 224/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2280 - val_loss: 0.6686
    Epoch 225/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2405 - val_loss: 0.6741
    Epoch 226/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1989 - val_loss: 0.6905
    Epoch 227/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2318 - val_loss: 0.6774
    Epoch 228/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2357 - val_loss: 0.7197
    Epoch 229/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2221 - val_loss: 0.8497
    Epoch 230/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2413 - val_loss: 0.8149
    Epoch 231/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2262 - val_loss: 0.9915
    Epoch 232/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2387 - val_loss: 0.8953
    Epoch 233/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2213 - val_loss: 1.0356
    Epoch 234/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2191 - val_loss: 1.2904
    Epoch 235/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2351 - val_loss: 1.2977
    Epoch 236/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2032 - val_loss: 1.3216
    Epoch 237/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2106 - val_loss: 1.2605
    Epoch 238/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2045 - val_loss: 1.3304
    Epoch 239/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2163 - val_loss: 1.3264
    Epoch 240/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2393 - val_loss: 1.2882
    Epoch 241/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.2030 - val_loss: 1.5280
    Epoch 242/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1926 - val_loss: 1.3382
    Epoch 243/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2082 - val_loss: 1.3047
    Epoch 244/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2171 - val_loss: 1.5007
    Epoch 245/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2219 - val_loss: 1.5019
    Epoch 246/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2125 - val_loss: 1.5163
    Epoch 247/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1974 - val_loss: 1.4775
    Epoch 248/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2148 - val_loss: 1.5083
    Epoch 249/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1774 - val_loss: 1.5277
    Epoch 250/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2053 - val_loss: 1.5302
    Epoch 251/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.2038 - val_loss: 1.5194
    Epoch 252/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1918 - val_loss: 1.5123
    Epoch 253/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1970 - val_loss: 1.4993
    Epoch 254/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1926 - val_loss: 1.5058
    Epoch 255/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1714 - val_loss: 1.4937
    Epoch 256/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1621 - val_loss: 1.5119
    Epoch 257/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1908 - val_loss: 1.5035
    Epoch 258/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.1852 - val_loss: 1.5124
    Epoch 259/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1721 - val_loss: 1.5054
    Epoch 260/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1725 - val_loss: 1.5187
    Epoch 261/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1760 - val_loss: 1.5175
    Epoch 262/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1694 - val_loss: 1.5056
    Epoch 263/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1572 - val_loss: 1.5032
    Epoch 264/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1600 - val_loss: 1.5108
    Epoch 265/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1484 - val_loss: 1.5037
    Epoch 266/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1482 - val_loss: 1.5226
    Epoch 267/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1499 - val_loss: 1.5047
    Epoch 268/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1449 - val_loss: 1.5124
    Epoch 269/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1777 - val_loss: 1.5155
    Epoch 270/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1583 - val_loss: 1.5153
    Epoch 271/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1309 - val_loss: 1.5194
    Epoch 272/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1481 - val_loss: 1.5062
    Epoch 273/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1306 - val_loss: 1.5244
    Epoch 274/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1375 - val_loss: 1.5182
    Epoch 275/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1289 - val_loss: 1.5123
    Epoch 276/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1195 - val_loss: 1.5282
    Epoch 277/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1114 - val_loss: 1.5170
    Epoch 278/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1244 - val_loss: 1.5256
    Epoch 279/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1386 - val_loss: 1.5231
    Epoch 280/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1073 - val_loss: 1.5384
    Epoch 281/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1142 - val_loss: 1.5358
    Epoch 282/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1068 - val_loss: 1.5375
    Epoch 283/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1060 - val_loss: 1.5370
    Epoch 284/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1063 - val_loss: 1.5273
    Epoch 285/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.1123 - val_loss: 1.5396
    Epoch 286/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0829 - val_loss: 1.5223
    Epoch 287/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0934 - val_loss: 1.5283
    Epoch 288/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0884 - val_loss: 1.5327
    Epoch 289/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0878 - val_loss: 1.5365
    Epoch 290/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0978 - val_loss: 1.5358
    Epoch 291/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0817 - val_loss: 1.5520
    Epoch 292/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0782 - val_loss: 1.5411
    Epoch 293/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0960 - val_loss: 1.5508
    Epoch 294/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0747 - val_loss: 1.5435
    Epoch 295/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0777 - val_loss: 1.5534
    Epoch 296/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0774 - val_loss: 1.5629
    Epoch 297/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0761 - val_loss: 1.5770
    Epoch 298/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0761 - val_loss: 1.5423
    Epoch 299/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0684 - val_loss: 1.5712
    Epoch 300/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0837 - val_loss: 1.5497
    Epoch 301/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0605 - val_loss: 1.5580
    Epoch 302/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0599 - val_loss: 1.5594
    Epoch 303/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0565 - val_loss: 1.5607
    Epoch 304/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0590 - val_loss: 1.5613
    Epoch 305/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0513 - val_loss: 1.5644
    Epoch 306/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0541 - val_loss: 1.5605
    Epoch 307/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0567 - val_loss: 1.5646
    Epoch 308/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0521 - val_loss: 1.5788
    Epoch 309/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0603 - val_loss: 1.5760
    Epoch 310/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0522 - val_loss: 1.5680
    Epoch 311/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0581 - val_loss: 1.5650
    Epoch 312/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0388 - val_loss: 1.5726
    Epoch 313/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0426 - val_loss: 1.5634
    Epoch 314/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0511 - val_loss: 1.5651
    Epoch 315/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0469 - val_loss: 1.5649
    Epoch 316/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0453 - val_loss: 1.5670
    Epoch 317/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0535 - val_loss: 1.5599
    Epoch 318/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0425 - val_loss: 1.5662
    Epoch 319/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0547 - val_loss: 1.5727
    Epoch 320/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0431 - val_loss: 1.5740
    Epoch 321/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0379 - val_loss: 1.5685
    Epoch 322/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0324 - val_loss: 1.5680
    Epoch 323/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0266 - val_loss: 1.5715
    Epoch 324/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0326 - val_loss: 1.5668
    Epoch 325/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0541 - val_loss: 1.5908
    Epoch 326/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0338 - val_loss: 1.5660
    Epoch 327/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0338 - val_loss: 1.5752
    Epoch 328/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0382 - val_loss: 1.5741
    Epoch 329/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0274 - val_loss: 1.5736
    Epoch 330/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0269 - val_loss: 1.5724
    Epoch 331/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0197 - val_loss: 1.5778
    Epoch 332/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0257 - val_loss: 1.5698
    Epoch 333/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0216 - val_loss: 1.5717
    Epoch 334/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0253 - val_loss: 1.5691
    Epoch 335/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0214 - val_loss: 1.5627
    Epoch 336/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0258 - val_loss: 1.5810
    Epoch 337/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0217 - val_loss: 1.5637
    Epoch 338/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0185 - val_loss: 1.5759
    Epoch 339/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0220 - val_loss: 1.5662
    Epoch 340/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0276 - val_loss: 1.5735
    Epoch 341/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0164 - val_loss: 1.5755
    Epoch 342/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0137 - val_loss: 1.5741
    Epoch 343/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0128 - val_loss: 1.5717
    Epoch 344/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0190 - val_loss: 1.5680
    Epoch 345/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0196 - val_loss: 1.5807
    Epoch 346/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0137 - val_loss: 1.5753
    Epoch 347/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0126 - val_loss: 1.5731
    Epoch 348/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0119 - val_loss: 1.5741
    Epoch 349/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0119 - val_loss: 1.5728
    Epoch 350/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0119 - val_loss: 1.5698
    Epoch 351/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0116 - val_loss: 1.5751
    Epoch 352/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0155 - val_loss: 1.5672
    Epoch 353/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0124 - val_loss: 1.5735
    Epoch 354/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0101 - val_loss: 1.5729
    Epoch 355/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0100 - val_loss: 1.5673
    Epoch 356/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0099 - val_loss: 1.5693
    Epoch 357/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0123 - val_loss: 1.5722
    Epoch 358/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0111 - val_loss: 1.5757
    Epoch 359/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0097 - val_loss: 1.5721
    Epoch 360/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0086 - val_loss: 1.5712
    Epoch 361/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0072 - val_loss: 1.5747
    Epoch 362/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0078 - val_loss: 1.5762
    Epoch 363/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0059 - val_loss: 1.5745
    Epoch 364/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0058 - val_loss: 1.5717
    Epoch 365/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0070 - val_loss: 1.5717
    Epoch 366/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0066 - val_loss: 1.5784
    Epoch 367/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0080 - val_loss: 1.5799
    Epoch 368/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0059 - val_loss: 1.5730
    Epoch 369/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0045 - val_loss: 1.5712
    Epoch 370/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0063 - val_loss: 1.5729
    Epoch 371/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0045 - val_loss: 1.5690
    Epoch 372/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0043 - val_loss: 1.5696
    Epoch 373/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0045 - val_loss: 1.5719
    Epoch 374/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0059 - val_loss: 1.5692
    Epoch 375/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0057 - val_loss: 1.5683
    Epoch 376/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0052 - val_loss: 1.5730
    Epoch 377/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0049 - val_loss: 1.5725
    Epoch 378/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0039 - val_loss: 1.5725
    Epoch 379/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0030 - val_loss: 1.5745
    Epoch 380/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0035 - val_loss: 1.5739
    Epoch 381/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0049 - val_loss: 1.5700
    Epoch 382/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5716
    Epoch 383/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0029 - val_loss: 1.5701
    Epoch 384/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0046 - val_loss: 1.5740
    Epoch 385/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0033 - val_loss: 1.5718
    Epoch 386/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0020 - val_loss: 1.5696
    Epoch 387/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0025 - val_loss: 1.5685
    Epoch 388/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0033 - val_loss: 1.5744
    Epoch 389/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0023 - val_loss: 1.5713
    Epoch 390/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0033 - val_loss: 1.5698
    Epoch 391/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0029 - val_loss: 1.5744
    Epoch 392/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5717
    Epoch 393/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0021 - val_loss: 1.5724
    Epoch 394/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5742
    Epoch 395/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5735
    Epoch 396/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0025 - val_loss: 1.5753
    Epoch 397/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0031 - val_loss: 1.5667
    Epoch 398/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0049 - val_loss: 1.5739
    Epoch 399/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0054 - val_loss: 1.5693
    Epoch 400/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0033 - val_loss: 1.5723
    Epoch 401/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5721
    Epoch 402/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0023 - val_loss: 1.5705
    Epoch 403/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5707
    Epoch 404/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5734
    Epoch 405/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5717
    Epoch 406/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5710
    Epoch 407/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0020 - val_loss: 1.5728
    Epoch 408/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5722
    Epoch 409/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0029 - val_loss: 1.5714
    Epoch 410/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5706
    Epoch 411/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5729
    Epoch 412/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5741
    Epoch 413/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5710
    Epoch 414/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5718
    Epoch 415/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5689
    Epoch 416/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5690
    Epoch 417/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0025 - val_loss: 1.5700
    Epoch 418/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5725
    Epoch 419/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5710
    Epoch 420/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5716
    Epoch 421/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0022 - val_loss: 1.5702
    Epoch 422/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5718
    Epoch 423/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5721
    Epoch 424/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.2322e-04 - val_loss: 1.5682
    Epoch 425/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5697
    Epoch 426/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5763
    Epoch 427/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5709
    Epoch 428/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.2430e-04 - val_loss: 1.5734
    Epoch 429/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5717
    Epoch 430/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0277e-04 - val_loss: 1.5732
    Epoch 431/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.7927e-04 - val_loss: 1.5720
    Epoch 432/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.5525e-04 - val_loss: 1.5698
    Epoch 433/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5655
    Epoch 434/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0024 - val_loss: 1.5766
    Epoch 435/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.3292e-04 - val_loss: 1.5764
    Epoch 436/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5747
    Epoch 437/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5745
    Epoch 438/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7633e-04 - val_loss: 1.5717
    Epoch 439/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5693
    Epoch 440/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5738
    Epoch 441/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5684
    Epoch 442/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5717
    Epoch 443/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0014 - val_loss: 1.5791
    Epoch 444/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5734
    Epoch 445/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0014 - val_loss: 1.5707
    Epoch 446/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5703
    Epoch 447/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5762
    Epoch 448/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5714
    Epoch 449/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5746
    Epoch 450/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5730
    Epoch 451/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5715
    Epoch 452/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0010 - val_loss: 1.5647
    Epoch 453/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5685
    Epoch 454/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5765
    Epoch 455/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5739
    Epoch 456/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.3389e-04 - val_loss: 1.5716
    Epoch 457/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5728
    Epoch 458/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.2634e-04 - val_loss: 1.5703
    Epoch 459/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5736
    Epoch 460/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0025 - val_loss: 1.5707
    Epoch 461/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5716
    Epoch 462/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5708
    Epoch 463/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5602
    Epoch 464/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5755
    Epoch 465/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5702
    Epoch 466/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5732
    Epoch 467/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5717
    Epoch 468/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0011 - val_loss: 1.5734
    Epoch 469/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5724
    Epoch 470/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5731
    Epoch 471/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5686
    Epoch 472/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5745
    Epoch 473/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5729
    Epoch 474/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5701
    Epoch 475/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0813e-04 - val_loss: 1.5772
    Epoch 476/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5747
    Epoch 477/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.2921e-04 - val_loss: 1.5685
    Epoch 478/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5730
    Epoch 479/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0031 - val_loss: 1.5718
    Epoch 480/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0047 - val_loss: 1.5762
    Epoch 481/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0022 - val_loss: 1.5641
    Epoch 482/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5708
    Epoch 483/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5639
    Epoch 484/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0036 - val_loss: 1.5718
    Epoch 485/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5615
    Epoch 486/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5774
    Epoch 487/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5719
    Epoch 488/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8950e-04 - val_loss: 1.5739
    Epoch 489/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.1599e-04 - val_loss: 1.5724
    Epoch 490/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0031 - val_loss: 1.5743
    Epoch 491/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5780
    Epoch 492/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0016 - val_loss: 1.5733
    Epoch 493/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5706
    Epoch 494/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.6705e-04 - val_loss: 1.5726
    Epoch 495/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.9965e-04 - val_loss: 1.5727
    Epoch 496/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.3142e-04 - val_loss: 1.5734
    Epoch 497/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5757
    Epoch 498/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5749
    Epoch 499/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5757
    Epoch 500/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.9507e-04 - val_loss: 1.5706
    Epoch 501/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5722
    Epoch 502/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5729
    Epoch 503/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5776
    Epoch 504/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0308e-04 - val_loss: 1.5704
    Epoch 505/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5733
    Epoch 506/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.1518e-04 - val_loss: 1.5745
    Epoch 507/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5761
    Epoch 508/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5698
    Epoch 509/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0043 - val_loss: 1.5693
    Epoch 510/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5694
    Epoch 511/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.2770e-04 - val_loss: 1.5736
    Epoch 512/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.7041e-04 - val_loss: 1.5726
    Epoch 513/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.6742e-04 - val_loss: 1.5731
    Epoch 514/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.1813e-04 - val_loss: 1.5707
    Epoch 515/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4870e-04 - val_loss: 1.5743
    Epoch 516/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0010 - val_loss: 1.5754
    Epoch 517/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0038 - val_loss: 1.5763
    Epoch 518/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0050 - val_loss: 1.5806
    Epoch 519/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0034 - val_loss: 1.5777
    Epoch 520/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5717
    Epoch 521/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5707
    Epoch 522/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9054e-04 - val_loss: 1.5737
    Epoch 523/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5703
    Epoch 524/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5722
    Epoch 525/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5725
    Epoch 526/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8906e-04 - val_loss: 1.5739
    Epoch 527/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.6124e-04 - val_loss: 1.5720
    Epoch 528/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5717
    Epoch 529/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5737
    Epoch 530/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.2710e-04 - val_loss: 1.5744
    Epoch 531/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4093e-04 - val_loss: 1.5702
    Epoch 532/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5692
    Epoch 533/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5670
    Epoch 534/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5729
    Epoch 535/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5759
    Epoch 536/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5770
    Epoch 537/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0043 - val_loss: 1.5651
    Epoch 538/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0021 - val_loss: 1.5663
    Epoch 539/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0013 - val_loss: 1.5720
    Epoch 540/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.6686e-04 - val_loss: 1.5726
    Epoch 541/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.5005e-04 - val_loss: 1.5753
    Epoch 542/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4181e-04 - val_loss: 1.5722
    Epoch 543/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4889e-04 - val_loss: 1.5742
    Epoch 544/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9701e-04 - val_loss: 1.5708
    Epoch 545/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8478e-04 - val_loss: 1.5693
    Epoch 546/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5627
    Epoch 547/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5726
    Epoch 548/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5639
    Epoch 549/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0029 - val_loss: 1.5699
    Epoch 550/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8660e-04 - val_loss: 1.5770
    Epoch 551/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5706
    Epoch 552/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5715
    Epoch 553/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.3096e-04 - val_loss: 1.5689
    Epoch 554/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5705
    Epoch 555/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5740
    Epoch 556/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.2071e-04 - val_loss: 1.5744
    Epoch 557/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5718
    Epoch 558/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8089e-04 - val_loss: 1.5752
    Epoch 559/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 9.3509e-04 - val_loss: 1.5722
    Epoch 560/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5757
    Epoch 561/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.5087e-04 - val_loss: 1.5743
    Epoch 562/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7556e-04 - val_loss: 1.5749
    Epoch 563/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9989e-04 - val_loss: 1.5714
    Epoch 564/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5741
    Epoch 565/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5712
    Epoch 566/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.3267e-04 - val_loss: 1.5774
    Epoch 567/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5730
    Epoch 568/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.7526e-04 - val_loss: 1.5738
    Epoch 569/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.7746e-04 - val_loss: 1.5736
    Epoch 570/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.1454e-04 - val_loss: 1.5726
    Epoch 571/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8850e-04 - val_loss: 1.5695
    Epoch 572/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5768
    Epoch 573/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5716
    Epoch 574/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.9341e-04 - val_loss: 1.5723
    Epoch 575/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0017e-04 - val_loss: 1.5698
    Epoch 576/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9531e-04 - val_loss: 1.5759
    Epoch 577/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5788
    Epoch 578/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0014 - val_loss: 1.5736
    Epoch 579/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0015 - val_loss: 1.5722
    Epoch 580/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5737
    Epoch 581/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.8272e-04 - val_loss: 1.5765
    Epoch 582/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5681
    Epoch 583/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5748
    Epoch 584/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0023 - val_loss: 1.5703
    Epoch 585/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5735
    Epoch 586/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5753
    Epoch 587/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0023 - val_loss: 1.5758
    Epoch 588/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5749
    Epoch 589/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 9.9254e-04 - val_loss: 1.5779
    Epoch 590/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5684
    Epoch 591/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.2657e-04 - val_loss: 1.5710
    Epoch 592/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5716
    Epoch 593/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.6131e-04 - val_loss: 1.5687
    Epoch 594/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.5744e-04 - val_loss: 1.5729
    Epoch 595/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8759e-04 - val_loss: 1.5733
    Epoch 596/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8773e-04 - val_loss: 1.5697
    Epoch 597/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5756
    Epoch 598/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0012 - val_loss: 1.5735
    Epoch 599/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5727
    Epoch 600/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4460e-04 - val_loss: 1.5712
    Epoch 601/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5702
    Epoch 602/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5651
    Epoch 603/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0030 - val_loss: 1.5761
    Epoch 604/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0050 - val_loss: 1.5737
    Epoch 605/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0050 - val_loss: 1.5580
    Epoch 606/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0031 - val_loss: 1.5769
    Epoch 607/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5767
    Epoch 608/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5760
    Epoch 609/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5700
    Epoch 610/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7373e-04 - val_loss: 1.5721
    Epoch 611/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.1098e-04 - val_loss: 1.5756
    Epoch 612/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5720
    Epoch 613/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5748
    Epoch 614/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5758
    Epoch 615/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.6906e-04 - val_loss: 1.5702
    Epoch 616/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5722
    Epoch 617/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5785
    Epoch 618/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0026 - val_loss: 1.5735
    Epoch 619/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5805
    Epoch 620/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5713
    Epoch 621/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9433e-04 - val_loss: 1.5724
    Epoch 622/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.5959e-04 - val_loss: 1.5726
    Epoch 623/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5737
    Epoch 624/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.2449e-04 - val_loss: 1.5712
    Epoch 625/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0017 - val_loss: 1.5723
    Epoch 626/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5725
    Epoch 627/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9001e-04 - val_loss: 1.5732
    Epoch 628/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5736
    Epoch 629/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5777
    Epoch 630/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0026 - val_loss: 1.5685
    Epoch 631/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5745
    Epoch 632/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5732
    Epoch 633/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.6300e-04 - val_loss: 1.5735
    Epoch 634/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.0751e-04 - val_loss: 1.5724
    Epoch 635/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0411e-04 - val_loss: 1.5688
    Epoch 636/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0018 - val_loss: 1.5664
    Epoch 637/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0079 - val_loss: 1.5652
    Epoch 638/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0034 - val_loss: 1.5800
    Epoch 639/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0044 - val_loss: 1.5783
    Epoch 640/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5690
    Epoch 641/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0025 - val_loss: 1.5745
    Epoch 642/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.4558e-04 - val_loss: 1.5687
    Epoch 643/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.2409e-04 - val_loss: 1.5729
    Epoch 644/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5723
    Epoch 645/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4712e-04 - val_loss: 1.5701
    Epoch 646/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.2349e-04 - val_loss: 1.5756
    Epoch 647/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5659
    Epoch 648/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8406e-04 - val_loss: 1.5668
    Epoch 649/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5708
    Epoch 650/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5738
    Epoch 651/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 9.6713e-04 - val_loss: 1.5699
    Epoch 652/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4860e-04 - val_loss: 1.5686
    Epoch 653/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7248e-04 - val_loss: 1.5706
    Epoch 654/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0013 - val_loss: 1.5739
    Epoch 655/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5709
    Epoch 656/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.1564e-04 - val_loss: 1.5720
    Epoch 657/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.0171e-04 - val_loss: 1.5697
    Epoch 658/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0494e-04 - val_loss: 1.5752
    Epoch 659/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5725
    Epoch 660/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.2722e-04 - val_loss: 1.5739
    Epoch 661/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.6649e-04 - val_loss: 1.5706
    Epoch 662/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8197e-04 - val_loss: 1.5717
    Epoch 663/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5669
    Epoch 664/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5686
    Epoch 665/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5803
    Epoch 666/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5720
    Epoch 667/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.5346e-04 - val_loss: 1.5735
    Epoch 668/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5748
    Epoch 669/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.4076e-04 - val_loss: 1.5720
    Epoch 670/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.6702e-04 - val_loss: 1.5700
    Epoch 671/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5722
    Epoch 672/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0018 - val_loss: 1.5709
    Epoch 673/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5725
    Epoch 674/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.6497e-04 - val_loss: 1.5678
    Epoch 675/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5667
    Epoch 676/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5736
    Epoch 677/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5630
    Epoch 678/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5721
    Epoch 679/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5710
    Epoch 680/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.8370e-04 - val_loss: 1.5725
    Epoch 681/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.0204e-04 - val_loss: 1.5726
    Epoch 682/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4986e-04 - val_loss: 1.5705
    Epoch 683/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0017e-04 - val_loss: 1.5713
    Epoch 684/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.0060e-04 - val_loss: 1.5704
    Epoch 685/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.6746e-04 - val_loss: 1.5678
    Epoch 686/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5726
    Epoch 687/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.1036e-04 - val_loss: 1.5722
    Epoch 688/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5724
    Epoch 689/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.2430e-04 - val_loss: 1.5692
    Epoch 690/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.3451e-04 - val_loss: 1.5703
    Epoch 691/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0078 - val_loss: 1.5762
    Epoch 692/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0099 - val_loss: 1.5745
    Epoch 693/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5742
    Epoch 694/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5692
    Epoch 695/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5774
    Epoch 696/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5698
    Epoch 697/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0010 - val_loss: 1.5703
    Epoch 698/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5690
    Epoch 699/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0032 - val_loss: 1.5743
    Epoch 700/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5695
    Epoch 701/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5695
    Epoch 702/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5704
    Epoch 703/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0038 - val_loss: 1.5578
    Epoch 704/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0050 - val_loss: 1.5688
    Epoch 705/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0023 - val_loss: 1.5742
    Epoch 706/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5693
    Epoch 707/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0012 - val_loss: 1.5714
    Epoch 708/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.1799e-04 - val_loss: 1.5724
    Epoch 709/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.2757e-04 - val_loss: 1.5724
    Epoch 710/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5716
    Epoch 711/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5736
    Epoch 712/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5705
    Epoch 713/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0014 - val_loss: 1.5623
    Epoch 714/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5718
    Epoch 715/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.4586e-04 - val_loss: 1.5713
    Epoch 716/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7290e-04 - val_loss: 1.5748
    Epoch 717/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5709
    Epoch 718/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0517e-04 - val_loss: 1.5708
    Epoch 719/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5742
    Epoch 720/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0030 - val_loss: 1.5675
    Epoch 721/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0029 - val_loss: 1.5807
    Epoch 722/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5748
    Epoch 723/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5692
    Epoch 724/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 6.7228e-04 - val_loss: 1.5711
    Epoch 725/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.4028e-04 - val_loss: 1.5682
    Epoch 726/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4556e-04 - val_loss: 1.5727
    Epoch 727/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9118e-04 - val_loss: 1.5740
    Epoch 728/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7442e-04 - val_loss: 1.5717
    Epoch 729/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.2097e-04 - val_loss: 1.5733
    Epoch 730/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5705
    Epoch 731/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4125e-04 - val_loss: 1.5738
    Epoch 732/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.5135e-04 - val_loss: 1.5740
    Epoch 733/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.0999e-04 - val_loss: 1.5734
    Epoch 734/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5795
    Epoch 735/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0020 - val_loss: 1.5712
    Epoch 736/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5660
    Epoch 737/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0025 - val_loss: 1.5705
    Epoch 738/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0081 - val_loss: 1.5716
    Epoch 739/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0065 - val_loss: 1.5662
    Epoch 740/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0162 - val_loss: 1.5856
    Epoch 741/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0108 - val_loss: 1.5651
    Epoch 742/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0053 - val_loss: 1.5714
    Epoch 743/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5748
    Epoch 744/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0023 - val_loss: 1.5746
    Epoch 745/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5735
    Epoch 746/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.6952e-04 - val_loss: 1.5713
    Epoch 747/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9297e-04 - val_loss: 1.5731
    Epoch 748/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5736
    Epoch 749/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.2883e-04 - val_loss: 1.5708
    Epoch 750/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.5183e-04 - val_loss: 1.5728
    Epoch 751/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.0429e-04 - val_loss: 1.5712
    Epoch 752/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.7793e-04 - val_loss: 1.5706
    Epoch 753/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 6.3346e-04 - val_loss: 1.5701
    Epoch 754/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.6664e-04 - val_loss: 1.5712
    Epoch 755/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8939e-04 - val_loss: 1.5724
    Epoch 756/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 8.0018e-04 - val_loss: 1.5716
    Epoch 757/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4573e-04 - val_loss: 1.5715
    Epoch 758/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.9756e-04 - val_loss: 1.5733
    Epoch 759/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.6044e-04 - val_loss: 1.5724
    Epoch 760/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.8492e-04 - val_loss: 1.5732
    Epoch 761/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.5484e-04 - val_loss: 1.5718
    Epoch 762/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.0668e-04 - val_loss: 1.5748
    Epoch 763/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0800e-04 - val_loss: 1.5710
    Epoch 764/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7006e-04 - val_loss: 1.5728
    Epoch 765/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5721
    Epoch 766/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.3817e-04 - val_loss: 1.5697
    Epoch 767/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.3667e-04 - val_loss: 1.5718
    Epoch 768/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7306e-04 - val_loss: 1.5683
    Epoch 769/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5733
    Epoch 770/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4519e-04 - val_loss: 1.5766
    Epoch 771/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0026 - val_loss: 1.5720
    Epoch 772/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5753
    Epoch 773/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5713
    Epoch 774/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5707
    Epoch 775/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8161e-04 - val_loss: 1.5703
    Epoch 776/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.1524e-04 - val_loss: 1.5761
    Epoch 777/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.5525e-04 - val_loss: 1.5636
    Epoch 778/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5719
    Epoch 779/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.6989e-04 - val_loss: 1.5708
    Epoch 780/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.9336e-04 - val_loss: 1.5721
    Epoch 781/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.1892e-04 - val_loss: 1.5729
    Epoch 782/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.2603e-04 - val_loss: 1.5737
    Epoch 783/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.1623e-04 - val_loss: 1.5740
    Epoch 784/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 5.4800e-04 - val_loss: 1.5710
    Epoch 785/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.1651e-04 - val_loss: 1.5753
    Epoch 786/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5756
    Epoch 787/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.5959e-04 - val_loss: 1.5724
    Epoch 788/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.8385e-04 - val_loss: 1.5710
    Epoch 789/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.4820e-04 - val_loss: 1.5750
    Epoch 790/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.0900e-04 - val_loss: 1.5715
    Epoch 791/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.5390e-04 - val_loss: 1.5731
    Epoch 792/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.7849e-04 - val_loss: 1.5704
    Epoch 793/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8136e-04 - val_loss: 1.5695
    Epoch 794/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0026 - val_loss: 1.5688
    Epoch 795/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5722
    Epoch 796/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5736
    Epoch 797/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4853e-04 - val_loss: 1.5713
    Epoch 798/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0013 - val_loss: 1.5665
    Epoch 799/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0022 - val_loss: 1.5696
    Epoch 800/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0020 - val_loss: 1.5700
    Epoch 801/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0025e-04 - val_loss: 1.5732
    Epoch 802/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5690
    Epoch 803/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5715
    Epoch 804/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 8.0840e-04 - val_loss: 1.5699
    Epoch 805/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.3573e-04 - val_loss: 1.5715
    Epoch 806/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.9581e-04 - val_loss: 1.5700
    Epoch 807/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.6622e-04 - val_loss: 1.5748
    Epoch 808/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.3199e-04 - val_loss: 1.5735
    Epoch 809/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9772e-04 - val_loss: 1.5735
    Epoch 810/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0887e-04 - val_loss: 1.5717
    Epoch 811/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5721
    Epoch 812/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5678
    Epoch 813/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.1872e-04 - val_loss: 1.5716
    Epoch 814/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9860e-04 - val_loss: 1.5729
    Epoch 815/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.3670e-04 - val_loss: 1.5708
    Epoch 816/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8596e-04 - val_loss: 1.5720
    Epoch 817/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7403e-04 - val_loss: 1.5689
    Epoch 818/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7842e-04 - val_loss: 1.5680
    Epoch 819/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0011 - val_loss: 1.5720
    Epoch 820/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5714
    Epoch 821/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0768e-04 - val_loss: 1.5724
    Epoch 822/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.2533e-04 - val_loss: 1.5739
    Epoch 823/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.8976e-04 - val_loss: 1.5721
    Epoch 824/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.5762e-04 - val_loss: 1.5702
    Epoch 825/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.0133e-04 - val_loss: 1.5731
    Epoch 826/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 5.3181e-04 - val_loss: 1.5709
    Epoch 827/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.4497e-04 - val_loss: 1.5750
    Epoch 828/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7849e-04 - val_loss: 1.5738
    Epoch 829/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.7596e-04 - val_loss: 1.5708
    Epoch 830/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.8880e-04 - val_loss: 1.5740
    Epoch 831/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8718e-04 - val_loss: 1.5692
    Epoch 832/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5738
    Epoch 833/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8593e-04 - val_loss: 1.5744
    Epoch 834/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5763
    Epoch 835/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0032 - val_loss: 1.5795
    Epoch 836/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0029 - val_loss: 1.5707
    Epoch 837/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0026 - val_loss: 1.5620
    Epoch 838/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0035 - val_loss: 1.5756
    Epoch 839/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5759
    Epoch 840/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5708
    Epoch 841/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5756
    Epoch 842/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5764
    Epoch 843/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0034 - val_loss: 1.5776
    Epoch 844/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0023 - val_loss: 1.5702
    Epoch 845/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0021 - val_loss: 1.5692
    Epoch 846/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5691
    Epoch 847/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4021e-04 - val_loss: 1.5742
    Epoch 848/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.8412e-04 - val_loss: 1.5716
    Epoch 849/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.5831e-04 - val_loss: 1.5704
    Epoch 850/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - loss: 8.1275e-04 - val_loss: 1.5734
    Epoch 851/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8932e-04 - val_loss: 1.5715
    Epoch 852/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5787
    Epoch 853/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5722
    Epoch 854/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5705
    Epoch 855/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5740
    Epoch 856/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5735
    Epoch 857/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0012 - val_loss: 1.5739
    Epoch 858/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5754
    Epoch 859/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.0526e-04 - val_loss: 1.5722
    Epoch 860/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.9614e-04 - val_loss: 1.5699
    Epoch 861/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.0508e-04 - val_loss: 1.5718
    Epoch 862/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0014 - val_loss: 1.5770
    Epoch 863/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9056e-04 - val_loss: 1.5714
    Epoch 864/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 9.4660e-04 - val_loss: 1.5704
    Epoch 865/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5691
    Epoch 866/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5716
    Epoch 867/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5705
    Epoch 868/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5720
    Epoch 869/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5696
    Epoch 870/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5716
    Epoch 871/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.8550e-04 - val_loss: 1.5694
    Epoch 872/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.2004e-04 - val_loss: 1.5742
    Epoch 873/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.2958e-04 - val_loss: 1.5719
    Epoch 874/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 5.5720e-04 - val_loss: 1.5719
    Epoch 875/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0020 - val_loss: 1.5679
    Epoch 876/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0030 - val_loss: 1.5734
    Epoch 877/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5704
    Epoch 878/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5692
    Epoch 879/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5710
    Epoch 880/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5722
    Epoch 881/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5727
    Epoch 882/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0016 - val_loss: 1.5733
    Epoch 883/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5720
    Epoch 884/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.1455e-04 - val_loss: 1.5722
    Epoch 885/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.2827e-04 - val_loss: 1.5693
    Epoch 886/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.4718e-04 - val_loss: 1.5712
    Epoch 887/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.0166e-04 - val_loss: 1.5703
    Epoch 888/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0011 - val_loss: 1.5692
    Epoch 889/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5711
    Epoch 890/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0054 - val_loss: 1.5687
    Epoch 891/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0040 - val_loss: 1.5644
    Epoch 892/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0038 - val_loss: 1.5749
    Epoch 893/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0024 - val_loss: 1.5730
    Epoch 894/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0015 - val_loss: 1.5751
    Epoch 895/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5722
    Epoch 896/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5746
    Epoch 897/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5727
    Epoch 898/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.3754e-04 - val_loss: 1.5734
    Epoch 899/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4952e-04 - val_loss: 1.5726
    Epoch 900/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0012 - val_loss: 1.5779
    Epoch 901/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5718
    Epoch 902/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.7466e-04 - val_loss: 1.5738
    Epoch 903/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.5508e-04 - val_loss: 1.5721
    Epoch 904/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.1342e-04 - val_loss: 1.5716
    Epoch 905/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.2071e-04 - val_loss: 1.5723
    Epoch 906/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 4.4814e-04 - val_loss: 1.5733
    Epoch 907/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.5665e-04 - val_loss: 1.5737
    Epoch 908/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.9103e-04 - val_loss: 1.5707
    Epoch 909/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8644e-04 - val_loss: 1.5703
    Epoch 910/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5693
    Epoch 911/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5743
    Epoch 912/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5714
    Epoch 913/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 6.2278e-04 - val_loss: 1.5738
    Epoch 914/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.4797e-04 - val_loss: 1.5735
    Epoch 915/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.9828e-04 - val_loss: 1.5726
    Epoch 916/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.6530e-04 - val_loss: 1.5701
    Epoch 917/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5696
    Epoch 918/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.7810e-04 - val_loss: 1.5704
    Epoch 919/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0014 - val_loss: 1.5716
    Epoch 920/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0028 - val_loss: 1.5804
    Epoch 921/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0020 - val_loss: 1.5694
    Epoch 922/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5767
    Epoch 923/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5730
    Epoch 924/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0023 - val_loss: 1.5734
    Epoch 925/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0015 - val_loss: 1.5683
    Epoch 926/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5749
    Epoch 927/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5655
    Epoch 928/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0028 - val_loss: 1.5710
    Epoch 929/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5722
    Epoch 930/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5764
    Epoch 931/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0033 - val_loss: 1.5771
    Epoch 932/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0016 - val_loss: 1.5717
    Epoch 933/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.3406e-04 - val_loss: 1.5710
    Epoch 934/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.2043e-04 - val_loss: 1.5711
    Epoch 935/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5729
    Epoch 936/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5725
    Epoch 937/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0016 - val_loss: 1.5775
    Epoch 938/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5758
    Epoch 939/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0013 - val_loss: 1.5720
    Epoch 940/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.6721e-04 - val_loss: 1.5713
    Epoch 941/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.5102e-04 - val_loss: 1.5750
    Epoch 942/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5708
    Epoch 943/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9523e-04 - val_loss: 1.5709
    Epoch 944/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.4250e-04 - val_loss: 1.5712
    Epoch 945/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 7.1155e-04 - val_loss: 1.5691
    Epoch 946/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5713
    Epoch 947/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.7726e-04 - val_loss: 1.5690
    Epoch 948/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.2420e-04 - val_loss: 1.5723
    Epoch 949/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.4004e-04 - val_loss: 1.5736
    Epoch 950/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4109e-04 - val_loss: 1.5737
    Epoch 951/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.6972e-04 - val_loss: 1.5724
    Epoch 952/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 6.2578e-04 - val_loss: 1.5721
    Epoch 953/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 6.0152e-04 - val_loss: 1.5671
    Epoch 954/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5700
    Epoch 955/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5737
    Epoch 956/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8531e-04 - val_loss: 1.5720
    Epoch 957/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5735
    Epoch 958/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 5.2182e-04 - val_loss: 1.5721
    Epoch 959/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.0011 - val_loss: 1.5748
    Epoch 960/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5735
    Epoch 961/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5734
    Epoch 962/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5734
    Epoch 963/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.2760e-04 - val_loss: 1.5692
    Epoch 964/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 7.1622e-04 - val_loss: 1.5686
    Epoch 965/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0010 - val_loss: 1.5729
    Epoch 966/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0027 - val_loss: 1.5732
    Epoch 967/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0026 - val_loss: 1.5790
    Epoch 968/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0019 - val_loss: 1.5766
    Epoch 969/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5712
    Epoch 970/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5695
    Epoch 971/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.1143e-04 - val_loss: 1.5710
    Epoch 972/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.8397e-04 - val_loss: 1.5719
    Epoch 973/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 8.9964e-04 - val_loss: 1.5693
    Epoch 974/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.7224e-04 - val_loss: 1.5707
    Epoch 975/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5702
    Epoch 976/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0024 - val_loss: 1.5667
    Epoch 977/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5720
    Epoch 978/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0012 - val_loss: 1.5728
    Epoch 979/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0018 - val_loss: 1.5717
    Epoch 980/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.9323e-04 - val_loss: 1.5668
    Epoch 981/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0017 - val_loss: 1.5701
    Epoch 982/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 0.0017 - val_loss: 1.5748
    Epoch 983/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5741
    Epoch 984/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.8484e-04 - val_loss: 1.5675
    Epoch 985/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0011 - val_loss: 1.5729
    Epoch 986/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5721
    Epoch 987/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 8.0056e-04 - val_loss: 1.5711
    Epoch 988/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0015 - val_loss: 1.5688
    Epoch 989/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5744
    Epoch 990/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0013 - val_loss: 1.5734
    Epoch 991/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0025 - val_loss: 1.5766
    Epoch 992/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0011 - val_loss: 1.5684
    Epoch 993/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - loss: 9.3645e-04 - val_loss: 1.5740
    Epoch 994/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 5.9425e-04 - val_loss: 1.5737
    Epoch 995/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 4.5531e-04 - val_loss: 1.5713
    Epoch 996/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 7.8820e-04 - val_loss: 1.5678
    Epoch 997/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0026 - val_loss: 1.5663
    Epoch 998/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0014 - val_loss: 1.5713
    Epoch 999/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 9.4681e-04 - val_loss: 1.5747
    Epoch 1000/1000
    [1m18/18[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.0010 - val_loss: 1.5733


We can use the returned history object to view the training and validation loss as a function of the training epochs. Below we can see that initially the validation loss increases. This is not a symptom of overfitting, but rather a local minimum associated with the training process. As training continues we can see that again the validation loss reduces and we end up with a model that has generalised to the training data.


```python
# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'val loss'])
plt.xlabel('Epochs')
plt.show()
```


    
![png](./Simple_sine_files/Simple_sine_18_0.png)
    


We can view the performance of our model on our data by plotting the models predictions. In general, training an accurate model will be a function of many variables including choice of activation function, number of layers, number of neurons and the structure of the training data. In this case we see some artifacts of the training process as the model is not 100% accurate. This could be improved by tweaking the aforementioned parameters.


```python
# use the model to predict the output
preds = quant_mod.predict(X)

# plot the training, validation and model predictions
plt.plot(train_X, train_Y, '.')
plt.plot(val_X, val_Y, 'r.')
plt.plot(X, preds)
plt.xlabel('Scaled input voltage (arb.)')
plt.ylabel('Scaled output voltage (arb.)')
plt.show()
```

    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step 



    
![png](./Simple_sine_files/Simple_sine_20_1.png)
    


# Exporting the model

We can export the model to a format that can be read by the Moku using the function `save_linn()`. This will export our trained model to a json file that can be read in by the Moku neural network instrument. 


```python
save_linn(quant_mod, input_channels=1, output_channels=1, file_name='simple_sine_model.linn')
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 109 cycles



```python

```
