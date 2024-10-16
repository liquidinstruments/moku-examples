```python
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
from copy import copy
from tqdm import tqdm

from moku.nn import LinnModel, save_linn
```

# Data Generation


```python
X = np.arange(0, np.pi*8, 0.01)
X_norm = np.linspace(-1, 1, len(X))

def gen_function(input_array):
    func_type = random.randint(1,3)
    num_cycles = random.randint(1,20)
    
    if func_type == 1:
        ### generate sine wave ###
        output_array = np.sin(num_cycles*input_array)
        answer = np.array((1,-1,-1))

    elif func_type == 2:
        ### generate square wave
        output_array = signal.square(num_cycles*input_array)
        answer = np.array((-1,1,-1))

    else:
        ### generate sawtooth wave
        output_array = signal.sawtooth(num_cycles*input_array)
        answer = np.array((-1,-1,1))

        
    return output_array, answer


Y_norm, Y_answer = gen_function(X)
plt.plot(X_norm, Y_norm)
plt.show()
```


    
![png](./Signal_ID_files/Signal_ID_2_0.png)
    



```python
# generate some training data
dr = 1000

training_data = np.zeros((dr, X_norm.size))

training_answers = np.zeros((dr, 3))

print(training_data.shape)
print(training_answers.shape)

for idx in tqdm(range(dr)):
    Y_norm, Y_answer = gen_function(X)
    training_data[idx, :] = Y_norm
    training_answers[idx,:] = Y_answer
```

    (1000, 2514)
    (1000, 3)


    100%|██████████| 1000/1000 [00:00<00:00, 17906.32it/s]



```python
# waveform batch size
bs = 32

N = X_norm.size
training_batched = []
answers_batched = []
for j in tqdm(range(training_data.shape[0])):
    for idx in range(0, N, bs):
        if idx+bs < N:
            training_batched.append(training_data[j, idx:idx+bs])
            answers_batched.append(training_answers[j])

training_batched = np.array(training_batched)
answers_batched = np.array(answers_batched)
print(training_batched.shape)
print(answers_batched.shape)

plt.plot(training_batched[20000,:])
print(answers_batched[20000,:])
```

    100%|██████████| 1000/1000 [00:00<00:00, 31139.27it/s]

    (78000, 32)
    (78000, 3)
    [ 1. -1. -1.]


    



    
![png](./Signal_ID_files/Signal_ID_4_3.png)
    


# Model Definitions


```python
# create the quantised model object and set the training data
quant_mod = LinnModel()
quant_mod.set_training_data(training_inputs=training_batched, training_outputs=answers_batched)
```


```python
# model definition for an autoencoder
model_definition = [(32, 'tanh'), (64, 'tanh'), (32, 'tanh'), (3, 'linear')]
# build the model
quant_mod.construct_model(model_definition, show_summary=True)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">1,056</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_3             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,112</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_4             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_5             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">99</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_clip_layer_6             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">OutputClipLayer</span>)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,347</span> (20.89 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,347</span> (20.89 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
history = quant_mod.fit_model(epochs=1000, es_config={'pateience': 20}, validation_split=0.1)
```

    Value for monitor missing. Using default:val_loss.
    Value for patience missing. Using default:5.
    Value for restore missing. Using default:False.


    Epoch 1/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 491us/step - loss: 0.0328 - val_loss: 0.0241
    Epoch 2/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 536us/step - loss: 0.0320 - val_loss: 0.0116
    Epoch 3/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 531us/step - loss: 0.0311 - val_loss: 0.0153
    Epoch 4/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 469us/step - loss: 0.0296 - val_loss: 0.0224
    Epoch 5/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 538us/step - loss: 0.0326 - val_loss: 0.0166
    Epoch 6/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 468us/step - loss: 0.0278 - val_loss: 0.0073
    Epoch 7/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 466us/step - loss: 0.0331 - val_loss: 0.0313
    Epoch 8/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 474us/step - loss: 0.0274 - val_loss: 0.0177
    Epoch 9/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 468us/step - loss: 0.0240 - val_loss: 0.0070
    Epoch 10/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 470us/step - loss: 0.0281 - val_loss: 0.0200
    Epoch 11/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 492us/step - loss: 0.0283 - val_loss: 0.0230
    Epoch 12/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 471us/step - loss: 0.0267 - val_loss: 0.0077
    Epoch 13/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 464us/step - loss: 0.0278 - val_loss: 0.0204
    Epoch 14/1000
    [1m2194/2194[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 469us/step - loss: 0.0238 - val_loss: 0.0082



```python
# plot the losses
plt.semilogy(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()
```


    
![png](./Signal_ID_files/Signal_ID_9_0.png)
    



```python
Y_norm, Y_answer = gen_function(X)

preds = quant_mod.predict(Y_norm[0:32].reshape(1,-1), scale=False, unscale_output=False)
print(Y_answer)
print(preds[0])
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 10ms/step
    [ 1 -1 -1]
    [ 1. -1. -1.]



```python
quant_mod.model.save('siginal_id.keras')
save_linn(quant_mod.model, input_channels=1, output_channels=3, file_name='signa_id.linn')
```

    Skipping layer 0 with type <class 'keras.src.layers.core.input_layer.InputLayer'>
    Skipping layer 2 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 4 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 6 with type <class 'moku.nn._linn.OutputClipLayer'>
    Skipping layer 8 with type <class 'moku.nn._linn.OutputClipLayer'>
    Network latency approx. 172 cycles



```python

```
