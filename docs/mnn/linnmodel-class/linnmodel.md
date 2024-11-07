# LinnModel class methods

The LinnModel class contains methods to construct, train and output a Liquid Instruments Neural Network (.linn) file that can be loaded into a Moku Neural Network instrument. These design elements are defined below, and can be used in your design like this:

```python
linn_model = LinnModel()
```


## set_training_data method

```python
Model.set_training_data(
    training_inputs=np.ndarray, 
    training_outputs=np.ndarray, 
    scale=bool, 
    input_data_boundary=Tuple[list], 
    output_data_boundary=Tuple[list]
)
```

Set the training data to be used by the model and automatically scale them to use the full dynamic range if specified. Scaling will ensure the model inputs and outputs are in the domain [-1, 1]. Call this before constructing a model to assign the data dimensions in construction.

### Parameters
- **training_inputs:** numpy.ndarray of input training data which should match the model dimensions.
- **training_outputs:** numpy.ndarray of output training data which should match the model dimensions.
- **scale:** automatically scale the data to fit in the bounds.
- **input_data_boundary:** tuple of the input data boundaries that should be used for scaling.
- **output_data_boundary:** tuple of the output data boundaries that should be used for scaling.

### Returns

None.

## construct_model method

```python
Model.construct_model(
        layer_definition=list,
        optimizer="adam",
        loss="mse",
        metrics=(),
)
```

Construct the model to be used by the rest of the functions in this class.

The `layer_definition` should include the output layer with an appropriate activation function, however the input layer is inferred from the training data set. For this reason, `set_training_data` should be called before `construct_model`.


### Parameters
- **layer_definition:** a list of tuples of the form `(layer_width, activation)` which defines the model. `(layer_width,)` can be used to signify a linear (identity) activation function.
- **optimizer:** optimizer fed to the keras compile option.
- **loss:** loss function fed to the keras compile option.
- **metrics:** metrics for the model to track during training.

### Returns

None.


## fit_model method

```python
Model.fit_model(
    epochs=int, 
    es_config=Optional[dict], 
    validation_split=float, 
    validation_data=Optional[tuple], 
    **keras_kwargs=Optional[Any]
)
```

Fit the model according to the data that is stored in the training inputs and outputs. `set_training_data` and `construct_model` must be called prior to calling `fit_model` to set the model parameters.

### Parameters
- **epochs:** (int) number of epochs to train for.
- **es_config:** configuration dictionary for the early stopping callback to stop training when a monitored metric has stopped improving. See [keras.callbacks.EarlyStopping()](https://keras.io/api/callbacks/early_stopping/) for more information.
- **validation_split:** (float) used to define the validation split.
- **validation_data:** validation data in a tuple of form (inputs, outputs).
- **keras_kwargs:** keyword args to pass to the keras `fit` function.

### Returns

History object from the keras `fit` function.


## predict method

```python
Model.predict(
    inputs=np.ndarray, 
    scale=Optional[bool], 
    unscale_output=Optional[bool], 
    **keras_kwargs=Optional[Any]
)
```

Generates model predictions for given inputs, with optional scaling based on training settings. Must be called after the model has been `fit`.

### Parameters
- **inputs:** The input data to be run through the model.
- **scale:** Whether the data should be scaled at the input to match the training data. Passing `None` (the default) will scale the data for prediction if and only if the training data was also scaled.
- **unscale_output:** Whether the data should be unscaled at the output to match the training data. Passing `None` (the default) will scale the prediction results if and only if the training results were also
- **keras_kwargs:** parameters to be passed to thhe keras predict function if needed.

### Returns

The model predictions.


## get_linn method

```python
get_linn( 
    model=Union[LinnModel, keras.models.Model], 
    input_channels=int, 
    output_channels=int, 
    **kwargs=any,
)
```

Converts a `LinnModel` into the `.linn` format required for execution on the Moku Neural Network Instrument. This function will also work with compatible Keras models if configured according to `LinnModel` standards.

*Serial mode:* When `input_channels` = 1 and the model expects multiple inputs (layer_definition inputs > 1), data will be fed as a sliding time window (e.g., with 32 input neurons, a 32-sample window is provided).

*Parallel mode:* If `input_channels` > 1, the number of inputs to the model must match `input_channels`, providing simultaneous samples from each instrument input.



### Arguments
- **model:** (keras.models.Model) The 'LinnModel' instance or a compatible Keras model.
- **input_channels:** An integer of the number of instrument inputs to connect to the network. Determines processing mode (serial or parallel) based on the ratio between 'input_channels' and the number of input neurons in the model.
- **output_channels:** An integer of the number of instrument outputs to connect to the network. Determines processing mode (serial or parallel) based on the ratio between 'output_channels' and the number of output neurons in the model.
- **kwargs:** (Optional).
    - **output_mapping:** (list): A list of integers that selects which output neurons should be used as the final output of the network. This is useful when training a network which produces multiple outputs during training, but when you only need a small number of values as outputs from the instrument. See the [Autoencoder example](/mnn/examples/Autoencoder), where the network is trained to produce 32 de-noised outputs but the final output in hardware is only the last value.

### Returns

A dict of network parameters suitable for loading in to the Neural Network instrument, or serializing as JSON to a .linn file.

## save_linn method

```python
save_linn( 
    model=Union[LinnModel, keras.models.Model], 
    input_channels=int, 
    output_channels=int, 
    file_name=str,
    **kwargs=any,
)
```

Converts a Keras model which is suitable for execution on the Moku Neural Network Instrument into the `.linn` format and saves it to a file suitable for loading in the Moku app. This is similar to `get_linn` but saves, insead of returns the data structure.


### Arguments
- **model:** (keras.models.Model) The 'LinnModel' instance or a compatible Keras model.
- **input_channels:** An integer of the number of instrument inputs to connect to the network. Determines processing mode (serial or parallel) based on the ratio between 'input_channels' and the number of input neurons in the model.
- **output_channels:** An integer of the number of instrument outputs to connect to the network. Determines processing mode (serial or parallel) based on the ratio between 'output_channels' and the number of output neurons in the model.
- **file_name:** Name of output `.linn` file, requires `.linn` extension.
- **kwargs:** (Optional).
    - **output_mapping:** (list): A list of integers that selects which output neurons should be used as the final output of the network. This is useful when training a network which produces multiple outputs during training, but when you only need a small number of values as outputs from the instrument. See the [Autoencoder example](/mnn/examples/Autoencoder), where the network is trained to produce 32 de-noised outputs but the final output in hardware is only the last value.

### Returns

None. 
