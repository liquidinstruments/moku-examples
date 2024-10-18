# Quantum emitter example

<example-actions directory="neural-network" filename="Emitter_control.ipynb"/>

In this example we will demonstrate an example of a neural network based control scheme for pumping a quantum emitter. We will simulate the overlap of an incident optical mode on with the preferred mode of the emitter. Note that since this example is illustrative we will not perform a full quantum treatment of the problem, but instead assume that sufficient overlap of the modes implies good mode matching and therefore, efficient pumping of the emitter.


```python
# import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm
import tensorflow as tf128

from moku.nn import LinnModel, save_linn

from emitter_simulator import QuantumEmitter

# set the seed for repeatability
np.random.seed(7)
tf128.random.set_seed(7)
```

The examples included in the `nn` package will allow us to simulate the emitter and the incident mode. We simulate this by assuming a Gaussian beam profile for both the emitter and incident mode. To simplify the problem, we will assume a fixed mode size and assume the only perturbations to the system are translations of the beam in the x-y plane and pointing errors with angles defined between the z-x and z-y axes. A diagrammatic representation of this is shown below:


We can plot the profile of the field as a function of x and y. At `z=0` there beam converges to the waist size but the radius of curvature becomes infinite, so instead we plot as $z \rightarrow 0$. We should a Gaussian beam with a 1/e beam diameter equal to the waist we defined.


```python
# define a space over which to plot, in this case 10um x 10um
x = np.linspace(-10e-6, 10e-6, 100)
X, Y = np.meshgrid(x, x, indexing='ij')

# define the emitter so we can simulate it
qe_sim = QuantumEmitter(wavelength=780e-9, waist=5e-6)
qe_sim.set_XY(X, Y)

# get the observable beam intensity
intensity = np.abs(qe_sim.E(X, Y, 1e-10))**2
plt.imshow(intensity, extent=[-10,10,-10,10])
plt.xlabel('X axis ($\mu$m)')
plt.ylabel('Y axis ($\mu$m)')
plt.colorbar()
plt.show()
```

    <>:12: SyntaxWarning: invalid escape sequence '\m'
    <>:13: SyntaxWarning: invalid escape sequence '\m'
    <>:12: SyntaxWarning: invalid escape sequence '\m'
    <>:13: SyntaxWarning: invalid escape sequence '\m'
    /var/folders/18/dqls73yj2sx1bdtnmn2y5glw0000gn/T/ipykernel_95738/1530026410.py:12: SyntaxWarning: invalid escape sequence '\m'
      plt.xlabel('X axis ($\mu$m)')
    /var/folders/18/dqls73yj2sx1bdtnmn2y5glw0000gn/T/ipykernel_95738/1530026410.py:13: SyntaxWarning: invalid escape sequence '\m'
      plt.ylabel('Y axis ($\mu$m)')



    
![png](./Emitter_control_files/Emitter_control_6_1.png)
    


To simulate misalignment of the input beam we wish for our control parameters to perform a random walk over time. To do this we will first define a random walk function which uses the `step_size` argument to determine the variance of the jump at each point. The `input_array` is expected to the series of time points over which the walk is generated. The random walk will always stay within the domain `[-1, 1]` so that we may scale the walk to the relevant parameter bounds.


```python
def random_walk(step_size, input_array, random_start=False):
    # start at 0 or some value in the domain
    if random_start:
        running_value = np.random.uniform(-1,1,1)[0]
    else:
        running_value = 0

    # generate an array of random steps
    output_array = np.random.normal(0, step_size, (input_array.size, 1))
    output_array[0] = running_value   # set the initial value of the walk  

    # for each item determine add the previous position to the walk and clip with the bounds
    for idx in range(output_array.shape[0]):
        if idx != 0:
            output_array[idx] = np.clip(output_array[idx] + output_array[idx - 1], -1, 1)

    return output_array
```

We can see the kind of random walk we are generating and gauge how severe we would like our driift to be.


```python
# generate a time base and plot random walks for different step sizes
T = np.linspace(0,1,1000)

# plot different step sizes for comparison
steps = [0.1, 0.01, 0.001]
for step_size in steps:
    walk = random_walk(step_size, T)
    plt.plot(T, walk)

plt.legend(steps)
plt.xlabel('Time (arb.)')
plt.ylabel('Walk position (arb.)')
plt.show()
```


    
![png](./Emitter_control_files/Emitter_control_10_0.png)
    


# Simulation example

We now will run our simulation to generate some training data which will be used to teach the model what corrections we need to make. The model will map observed difference in counts from each modulation to the the best action to perform at a given time step, effectively performing a single shot correction. Since we have centered the target beam at the coordinates (0, 0), and with 0 angle with respect to the z-axis, the ideal correction to perform is simply the negative of our current state. Note that the neural network model does not have access to the absolute position of the beam within the space, but rather is inferring the best action for a set of observables, in this case the difference in counts with each respective modulation. 

We start by defining a random walk for each of the control parameters:


```python
# time base over which to simulate
T = np.linspace(0, 1, 2500)

# generate the random walks
X_offset= random_walk(0.1, T)
Y_offset = random_walk(0.1, T)
X_angle = random_walk(0.1, T)
Y_angle = random_walk(0.1, T)

# set the scale of each variable accordingly These are scaled to bounds for which our simulation makes sense.
X_offset *= 4e-6
Y_offset *= 4e-6
X_angle += 1
X_angle *= 5
Y_angle += 1
Y_angle *= 5
```

Next we will simulate the random walk of each parameter at a given timestep while recording the counts without the correction, as well as the differential counts as a function of the modulations.


```python
counts = np.zeros(T.size)
mod_counts = np.zeros((T.size, 4))
corrections = np.zeros((T.size, 4))

for i in tqdm(range(T.size)):
    # get the new beam params
    x_off = X_offset[i]
    y_off = Y_offset[i]
    x_ang = X_angle[i] * np.pi / 180
    y_ang = Y_angle[i] * np.pi / 180

    # calculate the shears and new angles
    offsets = (x_off, y_off)
    shears = (qe_sim.new_scale(np.pi/2 - x_ang),qe_sim.new_scale(np.pi/2 - y_ang))
    angles = [np.pi/2 - x_ang, np.pi/2 - y_ang]

    # run the step, save values
    all_counts = qe_sim.time_step(offsets, shears, angles)
    base_count = qe_sim.get_counts()
    mod_counts[i] = all_counts
    counts[i] = base_count
    corrections[i] = np.array([0,0,1,1]) - np.array([*offsets, *shears]).flatten()
```

    100%|██████████| 2500/2500 [00:03<00:00, 707.08it/s]


It is important to note that we are able to generate a training data set in this way as we can simulate the system and therefore know the ideal correction should be. It may also be able to construct a training set from measured data if the ideal correction can also be inferred or calculated. In the abscence of this, however, it may still be possible to construct a training set using reinforcement learning methods such as q-learning. An in-depth discussion of these methods is outside the scope of this tutorial, but instead may be found here: [Goodfellow-Ch14](https://www.deeplearningbook.org/contents/autoencoders.html)

## Model definition and training

Having created our training set we can now train the model to perform the corrections. We will use a feed-forward multi-layer perceptron for this with 4 intermediate layers. The intermediate layers will all have a ReLU activation function. 


```python
#constructing  the model
quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=mod_counts, training_outputs=corrections)

# define the five layer model
model_definition = [(100, 'relu'),(100, 'relu'), (64, 'relu'), (64, 'relu'), (4,'linear')]
quant_mod.construct_model(model_definition, show_summary=True)
```

```
**Model: "functional"**

| Layer (type)                          | Output Shape           | Param #       |
|---------------------------------------|------------------------|---------------|
| input_layer (InputLayer)              | (None, 4)              | 0             |
| dense (Dense)                         | (None, 100)            | 500           |
| output_clip_layer (OutputClipLayer)   | (None, 100)            | 0             |
| dense_1 (Dense)                       | (None, 100)            | 10,100        |
| output_clip_layer_1 (OutputClipLayer) | (None, 100)            | 0             |
| dense_2 (Dense)                       | (None, 64)             | 6,464         |
| output_clip_layer_2 (OutputClipLayer) | (None, 64)             | 0             |
| dense_3 (Dense)                       | (None, 64)             | 4,160         |
| output_clip_layer_3 (OutputClipLayer) | (None, 64)             | 0             |
| dense_4 (Dense)                       | (None, 4)              | 260           |
| output_clip_layer_4 (OutputClipLayer) | (None, 4)              | 0             |

**Total params:** 21,484 (83.92 KB)
**Trainable params:** 21,484 (83.92 KB)
**Non-trainable params:** 0 (0.00 B)

```



We will train this model up to 500 epochs using early-stopping to ensure we do not overfit. We will also us a 10% validation split so that we may monitor our progress. On completion of the training we can view the training history to see how well the model is performing.


```python
history = quant_mod.fit_model(epochs=500, es_config={'patience':50, 'restore':True}, validation_split=0.1)
```

```
Value for monitor missing. Using default:val_loss.

Epoch 1/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2640 - val_loss: 0.2301
Epoch 2/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 0.0747 - val_loss: 0.2169
Epoch 3/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0585 - val_loss: 0.2146
Epoch 4/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0535 - val_loss: 0.2130
Epoch 5/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0507 - val_loss: 0.2152
Epoch 6/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.0485 - val_loss: 0.2160
.
.
.
.
.
.

Epoch 203/500
[1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.0061 - val_loss: 0.1128
```


```python
# plot the loss and validation loss as a function of epoch to see how our training went.
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()
```


    
![png](./Emitter_control_files/Emitter_control_22_0.png)
    


Now that we have a trained model we should simulate the system again while applying the corrections to view the new performance of the model. We can compare these to the uncorrected results that we calculated earlier. 


```python
preds = quant_mod.predict(mod_counts, scale=True, unscale_output=True)
counts_cor = np.zeros(T.size)

for i in tqdm(range(T.size)):
    # get the new beam params
    x_off = X_offset[i]
    y_off = Y_offset[i]
    x_ang = X_angle[i] * np.pi / 180
    y_ang = Y_angle[i] * np.pi / 180

    # calculate the shears and new angles
    offsets = (x_off, y_off)
    shears = (qe_sim.new_scale(np.pi/2 - x_ang),qe_sim.new_scale(np.pi/2 - y_ang))
    angles = [np.pi/2 - x_ang, np.pi/2 - y_ang]

    # run the step before perofming the correction
    _ = qe_sim.time_step(offsets, shears, angles)
    
    _ = qe_sim.time_step((offsets[0]+preds[i][0],offsets[1]+preds[i][1]), (shears[0]+preds[i][2],shears[1]+preds[i][3]), angles)
    cnt_cor = qe_sim.get_counts()
    counts_cor[i] = cnt_cor
```

    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step


    100%|██████████| 2500/2500 [00:06<00:00, 363.75it/s]


Finally we can plot and calculate the our average fidelity compared to our random walk.


```python
print("Corrected [mean:std]:\t [%.2f, %.2f]" % (np.mean(counts_cor / 2**16), np.std(counts_cor / 2**16)))
print("Uncorrected [mean:std]:\t [%.2f, %.2f]" % (np.mean(counts / 2**16), np.std(counts / 2**16)))

plt.figure(figsize=(15,5))
plt.plot(counts / 2**16)
plt.plot(counts_cor / 2**16)
plt.legend(['Uncorrected', 'Corrected'])
plt.xlabel('Time (arb.)')
plt.ylabel('Normalised counts')
plt.show()
```

    Corrected [mean:std]:	 [0.99, 0.02]
    Uncorrected [mean:std]:	 [0.58, 0.17]



    
![png](./Emitter_control_files/Emitter_control_26_1.png)
    

