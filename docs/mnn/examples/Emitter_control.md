# Quantum emitter example

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

<div>
<img src="attachment:20f845a2-08a3-4cad-9e7f-c3b87b46ceff.png" width="500"/>
</div>

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


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │           <span style="color: #00af00; text-decoration-color: #00af00">500</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">10,100</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_1             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">6,464</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_2             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">4,160</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_3             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">260</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_4             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,484</span> (83.92 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,484</span> (83.92 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



We will train this model up to 500 epochs using early-stopping to ensure we do not overfit. We will also us a 10% validation split so that we may monitor our progress. On completion of the training we can view the training history to see how well the model is performing.


```python
history = quant_mod.fit_model(epochs=500, es_config={'patience':50, 'restore':True}, validation_split=0.1)
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
    Epoch 7/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0467 - val_loss: 0.2159
    Epoch 8/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0449 - val_loss: 0.2124
    Epoch 9/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0435 - val_loss: 0.2122
    Epoch 10/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0424 - val_loss: 0.2112
    Epoch 11/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0411 - val_loss: 0.2080
    Epoch 12/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0400 - val_loss: 0.2074
    Epoch 13/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0390 - val_loss: 0.2076
    Epoch 14/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0381 - val_loss: 0.2035
    Epoch 15/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0368 - val_loss: 0.2025
    Epoch 16/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0358 - val_loss: 0.1999
    Epoch 17/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0347 - val_loss: 0.1963
    Epoch 18/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0338 - val_loss: 0.1949
    Epoch 19/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 786us/step - loss: 0.0330 - val_loss: 0.1950
    Epoch 20/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 909us/step - loss: 0.0322 - val_loss: 0.1936
    Epoch 21/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 970us/step - loss: 0.0312 - val_loss: 0.1899
    Epoch 22/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 941us/step - loss: 0.0304 - val_loss: 0.1881
    Epoch 23/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 941us/step - loss: 0.0294 - val_loss: 0.1907
    Epoch 24/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 831us/step - loss: 0.0286 - val_loss: 0.1910
    Epoch 25/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 828us/step - loss: 0.0277 - val_loss: 0.1888
    Epoch 26/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 980us/step - loss: 0.0270 - val_loss: 0.1903
    Epoch 27/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0262 - val_loss: 0.1878
    Epoch 28/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 962us/step - loss: 0.0256 - val_loss: 0.1856
    Epoch 29/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 933us/step - loss: 0.0249 - val_loss: 0.1874
    Epoch 30/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 950us/step - loss: 0.0244 - val_loss: 0.1826
    Epoch 31/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 894us/step - loss: 0.0238 - val_loss: 0.1783
    Epoch 32/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 888us/step - loss: 0.0232 - val_loss: 0.1781
    Epoch 33/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 879us/step - loss: 0.0227 - val_loss: 0.1756
    Epoch 34/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 863us/step - loss: 0.0224 - val_loss: 0.1716
    Epoch 35/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 812us/step - loss: 0.0218 - val_loss: 0.1701
    Epoch 36/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 787us/step - loss: 0.0213 - val_loss: 0.1656
    Epoch 37/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 816us/step - loss: 0.0210 - val_loss: 0.1690
    Epoch 38/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 813us/step - loss: 0.0207 - val_loss: 0.1703
    Epoch 39/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 839us/step - loss: 0.0203 - val_loss: 0.1689
    Epoch 40/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 868us/step - loss: 0.0199 - val_loss: 0.1707
    Epoch 41/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 801us/step - loss: 0.0194 - val_loss: 0.1694
    Epoch 42/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 845us/step - loss: 0.0191 - val_loss: 0.1638
    Epoch 43/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 816us/step - loss: 0.0186 - val_loss: 0.1606
    Epoch 44/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.0184 - val_loss: 0.1602
    Epoch 45/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0180 - val_loss: 0.1539
    Epoch 46/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0177 - val_loss: 0.1525
    Epoch 47/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0171 - val_loss: 0.1504
    Epoch 48/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0169 - val_loss: 0.1488
    Epoch 49/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0166 - val_loss: 0.1465
    Epoch 50/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0163 - val_loss: 0.1449
    Epoch 51/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 856us/step - loss: 0.0161 - val_loss: 0.1460
    Epoch 52/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0158 - val_loss: 0.1445
    Epoch 53/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0154 - val_loss: 0.1445
    Epoch 54/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 735us/step - loss: 0.0154 - val_loss: 0.1461
    Epoch 55/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 0.0151 - val_loss: 0.1453
    Epoch 56/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 888us/step - loss: 0.0149 - val_loss: 0.1461
    Epoch 57/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 971us/step - loss: 0.0144 - val_loss: 0.1376
    Epoch 58/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 981us/step - loss: 0.0144 - val_loss: 0.1418
    Epoch 59/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 825us/step - loss: 0.0143 - val_loss: 0.1435
    Epoch 60/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 783us/step - loss: 0.0141 - val_loss: 0.1389
    Epoch 61/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0138 - val_loss: 0.1437
    Epoch 62/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0140 - val_loss: 0.1422
    Epoch 63/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0138 - val_loss: 0.1444
    Epoch 64/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0135 - val_loss: 0.1429
    Epoch 65/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0136 - val_loss: 0.1449
    Epoch 66/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0131 - val_loss: 0.1413
    Epoch 67/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 764us/step - loss: 0.0130 - val_loss: 0.1363
    Epoch 68/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0129 - val_loss: 0.1358
    Epoch 69/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0127 - val_loss: 0.1349
    Epoch 70/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0125 - val_loss: 0.1327
    Epoch 71/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0122 - val_loss: 0.1323
    Epoch 72/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.0124 - val_loss: 0.1270
    Epoch 73/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0122 - val_loss: 0.1269
    Epoch 74/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0120 - val_loss: 0.1282
    Epoch 75/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.0117 - val_loss: 0.1280
    Epoch 76/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0115 - val_loss: 0.1262
    Epoch 77/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 803us/step - loss: 0.0115 - val_loss: 0.1299
    Epoch 78/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.0112 - val_loss: 0.1243
    Epoch 79/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 770us/step - loss: 0.0109 - val_loss: 0.1224
    Epoch 80/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0108 - val_loss: 0.1240
    Epoch 81/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 777us/step - loss: 0.0107 - val_loss: 0.1279
    Epoch 82/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0109 - val_loss: 0.1285
    Epoch 83/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0107 - val_loss: 0.1271
    Epoch 84/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 775us/step - loss: 0.0109 - val_loss: 0.1245
    Epoch 85/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0105 - val_loss: 0.1272
    Epoch 86/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.0104 - val_loss: 0.1265
    Epoch 87/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.0103 - val_loss: 0.1226
    Epoch 88/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 737us/step - loss: 0.0103 - val_loss: 0.1301
    Epoch 89/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 761us/step - loss: 0.0102 - val_loss: 0.1242
    Epoch 90/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 743us/step - loss: 0.0102 - val_loss: 0.1252
    Epoch 91/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0103 - val_loss: 0.1257
    Epoch 92/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0100 - val_loss: 0.1268
    Epoch 93/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 824us/step - loss: 0.0100 - val_loss: 0.1266
    Epoch 94/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 762us/step - loss: 0.0098 - val_loss: 0.1224
    Epoch 95/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 794us/step - loss: 0.0099 - val_loss: 0.1229
    Epoch 96/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 778us/step - loss: 0.0099 - val_loss: 0.1275
    Epoch 97/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 790us/step - loss: 0.0094 - val_loss: 0.1269
    Epoch 98/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 789us/step - loss: 0.0095 - val_loss: 0.1229
    Epoch 99/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 822us/step - loss: 0.0097 - val_loss: 0.1309
    Epoch 100/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.0095 - val_loss: 0.1269
    Epoch 101/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 788us/step - loss: 0.0096 - val_loss: 0.1277
    Epoch 102/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0095 - val_loss: 0.1253
    Epoch 103/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0096 - val_loss: 0.1272
    Epoch 104/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0097 - val_loss: 0.1321
    Epoch 105/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0094 - val_loss: 0.1296
    Epoch 106/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0096 - val_loss: 0.1268
    Epoch 107/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0094 - val_loss: 0.1235
    Epoch 108/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0093 - val_loss: 0.1279
    Epoch 109/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0092 - val_loss: 0.1261
    Epoch 110/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0090 - val_loss: 0.1239
    Epoch 111/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 768us/step - loss: 0.0091 - val_loss: 0.1223
    Epoch 112/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0089 - val_loss: 0.1239
    Epoch 113/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0088 - val_loss: 0.1254
    Epoch 114/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 771us/step - loss: 0.0087 - val_loss: 0.1251
    Epoch 115/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0084 - val_loss: 0.1236
    Epoch 116/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0084 - val_loss: 0.1252
    Epoch 117/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0083 - val_loss: 0.1219
    Epoch 118/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0082 - val_loss: 0.1244
    Epoch 119/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0082 - val_loss: 0.1230
    Epoch 120/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0082 - val_loss: 0.1211
    Epoch 121/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 765us/step - loss: 0.0080 - val_loss: 0.1257
    Epoch 122/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0079 - val_loss: 0.1258
    Epoch 123/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0079 - val_loss: 0.1251
    Epoch 124/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.0079 - val_loss: 0.1208
    Epoch 125/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0078 - val_loss: 0.1228
    Epoch 126/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 806us/step - loss: 0.0077 - val_loss: 0.1197
    Epoch 127/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 759us/step - loss: 0.0077 - val_loss: 0.1166
    Epoch 128/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0077 - val_loss: 0.1201
    Epoch 129/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0076 - val_loss: 0.1225
    Epoch 130/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 755us/step - loss: 0.0077 - val_loss: 0.1209
    Epoch 131/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0075 - val_loss: 0.1157
    Epoch 132/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0074 - val_loss: 0.1160
    Epoch 133/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0077 - val_loss: 0.1187
    Epoch 134/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 766us/step - loss: 0.0075 - val_loss: 0.1171
    Epoch 135/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0079 - val_loss: 0.1172
    Epoch 136/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0075 - val_loss: 0.1170
    Epoch 137/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 0.0074 - val_loss: 0.1153
    Epoch 138/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0075 - val_loss: 0.1131
    Epoch 139/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0074 - val_loss: 0.1162
    Epoch 140/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0075 - val_loss: 0.1180
    Epoch 141/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0076 - val_loss: 0.1225
    Epoch 142/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 795us/step - loss: 0.0073 - val_loss: 0.1136
    Epoch 143/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0075 - val_loss: 0.1104
    Epoch 144/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 760us/step - loss: 0.0077 - val_loss: 0.1126
    Epoch 145/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0074 - val_loss: 0.1116
    Epoch 146/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0077 - val_loss: 0.1129
    Epoch 147/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0075 - val_loss: 0.1091
    Epoch 148/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 734us/step - loss: 0.0078 - val_loss: 0.1176
    Epoch 149/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0074 - val_loss: 0.1140
    Epoch 150/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0074 - val_loss: 0.1172
    Epoch 151/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0074 - val_loss: 0.1111
    Epoch 152/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0071 - val_loss: 0.1154
    Epoch 153/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0070 - val_loss: 0.1077
    Epoch 154/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0072 - val_loss: 0.1159
    Epoch 155/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 738us/step - loss: 0.0072 - val_loss: 0.1182
    Epoch 156/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0070 - val_loss: 0.1125
    Epoch 157/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 774us/step - loss: 0.0070 - val_loss: 0.1114
    Epoch 158/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.0073 - val_loss: 0.1197
    Epoch 159/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0076 - val_loss: 0.1188
    Epoch 160/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0072 - val_loss: 0.1217
    Epoch 161/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0073 - val_loss: 0.1185
    Epoch 162/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 772us/step - loss: 0.0074 - val_loss: 0.1170
    Epoch 163/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 809us/step - loss: 0.0076 - val_loss: 0.1218
    Epoch 164/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 874us/step - loss: 0.0072 - val_loss: 0.1189
    Epoch 165/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 816us/step - loss: 0.0072 - val_loss: 0.1167
    Epoch 166/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 805us/step - loss: 0.0071 - val_loss: 0.1301
    Epoch 167/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 827us/step - loss: 0.0071 - val_loss: 0.1233
    Epoch 168/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 773us/step - loss: 0.0071 - val_loss: 0.1215
    Epoch 169/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0070 - val_loss: 0.1231
    Epoch 170/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 742us/step - loss: 0.0068 - val_loss: 0.1167
    Epoch 171/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 741us/step - loss: 0.0068 - val_loss: 0.1209
    Epoch 172/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 758us/step - loss: 0.0071 - val_loss: 0.1191
    Epoch 173/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0066 - val_loss: 0.1111
    Epoch 174/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0067 - val_loss: 0.1169
    Epoch 175/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0065 - val_loss: 0.1174
    Epoch 176/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 748us/step - loss: 0.0068 - val_loss: 0.1165
    Epoch 177/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 756us/step - loss: 0.0064 - val_loss: 0.1206
    Epoch 178/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 753us/step - loss: 0.0066 - val_loss: 0.1159
    Epoch 179/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 740us/step - loss: 0.0066 - val_loss: 0.1173
    Epoch 180/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 754us/step - loss: 0.0065 - val_loss: 0.1116
    Epoch 181/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0067 - val_loss: 0.1191
    Epoch 182/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0065 - val_loss: 0.1147
    Epoch 183/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 749us/step - loss: 0.0064 - val_loss: 0.1162
    Epoch 184/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0063 - val_loss: 0.1166
    Epoch 185/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0064 - val_loss: 0.1129
    Epoch 186/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 787us/step - loss: 0.0062 - val_loss: 0.1105
    Epoch 187/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 736us/step - loss: 0.0063 - val_loss: 0.1192
    Epoch 188/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 793us/step - loss: 0.0063 - val_loss: 0.1189
    Epoch 189/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0064 - val_loss: 0.1212
    Epoch 190/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 747us/step - loss: 0.0062 - val_loss: 0.1136
    Epoch 191/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 751us/step - loss: 0.0062 - val_loss: 0.1159
    Epoch 192/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 745us/step - loss: 0.0063 - val_loss: 0.1197
    Epoch 193/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 752us/step - loss: 0.0062 - val_loss: 0.1181
    Epoch 194/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 750us/step - loss: 0.0063 - val_loss: 0.1158
    Epoch 195/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 739us/step - loss: 0.0061 - val_loss: 0.1141
    Epoch 196/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 729us/step - loss: 0.0059 - val_loss: 0.1156
    Epoch 197/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 733us/step - loss: 0.0062 - val_loss: 0.1189
    Epoch 198/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0064 - val_loss: 0.1181
    Epoch 199/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 730us/step - loss: 0.0063 - val_loss: 0.1163
    Epoch 200/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 744us/step - loss: 0.0062 - val_loss: 0.1171
    Epoch 201/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 746us/step - loss: 0.0061 - val_loss: 0.1177
    Epoch 202/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 757us/step - loss: 0.0062 - val_loss: 0.1161
    Epoch 203/500
    [1m71/71[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 763us/step - loss: 0.0061 - val_loss: 0.1128



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
    

