
TODO

1) FIX THESE ISSUES WITH eRF calculation:
	a) ensure that eRFs are calculated in the appropriate order (use a sorteddict)
	b) figure out why the "start" coordinate is fucked.
1) Pull from DB for visualizations
2) Evaluation script: Visualizations w/ LRP + plotting loss curves + plotting accuracy curves across multiple models.
3) Fix tests
4) Fix the contextual model implementation. A) Think about optimal bio CRF/eCRF sizes. B) Add the ability to automatically learn these from the data


## Project overview:

- configuration
	+ `config.py`
	+ `setup.py`
	+ `db/credentials.py`
- experiments (stored in db)
	+ `models/experiments.py`
		- parameters
			-datasets
				+ `encode_dataset.py`
			-model structures (different types of normalizations, etc)
				+ e.g., `models/structs/one_layer_conv_mlp/`
			-hyper parameters
- training and evaluation
	+ `main.py --experiment={experiment_name}`


## Preparing and executing experiments:

1. Adjust the file `config.py` to point to directories containing your images and where you intend to save data.
	a. Each image dataset should be in its own folder in the root directory `self.data_root` contained in the config.

2. Setup your python environment and install postgres if you haven't already.
	a. Execute the command: `python setup.py install`

3. Fix your credentials file.
	a. `cp db/credentials.py.template db/credentials.py` then edit the file to match your system and preferences.

4. Create datasets with `encode_dataset.py` 
	a. For each dataset you want to encode, create a class in `dataset_processing/` by following the `mnist.py` or `cifar.py` templates.
	b. To create a dataset: `python encode_dataset.py --dataset=mnist`

5. Create models.
	a. Add a folder with the kind of model you are creating. Follow the template of `models/structs/one_layer_conv_mlp/`.
	b. Coordinate the models you create with `models/experiments`. This script will allow you to build workers that run through any number of hyperparameter/model/dataset combinations of your experiment. For instance, if you want to measure the performance of models containing different kinds of normalization. Each model (same architecture but different normalizations) has a structure documented in `models/structs/one_layer_conv_mlp/`. The name of each model is added to the `one_layer_conv_mlp` method in `models/experiments.py` in the field "model_struct". Adding multiple entries in other fields, such as "dataset", will have the workers run through those combinations.
	c. See "Model construction" below for details on model construction

6. Populate database with experiments.
	a. Running with the example laid out in (3), populate your postgres DB with all possible experiment combinations: `python prepare_experiments.py --initialize --experiment_name=two_layer_conv_mlp`.
	b. If you don't want to delete you entire DB every time, omit the `--initialize` flag.
	c. I manually access the db with `psql contextual_DCN -h 127.0.0.1 -d contextual_DCN`.

7. Run models.
	a. Run a worker with `CUDA_VISIBLE_DEVICES=0 python main.py --experiment={experiment_name}`. The worker will continue until there are no more experiments to run in the DB.
	b. If you forgot what experiments you've created use `python main.py --list_experiments` to list all in the DB.
	c. Every worker will save tensorflow model checkpoints, tensorflow summaries that can be viewed in tensorboard, update the DB with its progress, and after finishing, will produce plots of the training performance.

## Model construction:

- Models are constructed similarly to caffe, as lists of layers.

- You can manually specify an "output" layer (HOW?). Otherwise a FC layer will automatically be added that maps your final tower activities to a layer with O computational units, where O = the number of categories in your dataset (this will fail on regression tasks).

```
        conv_tower = [
		{  # Note that each attribute is in a list. This is because for the case of resnet you must specify multiple attributes per layer (see below for example).
	        'layers': ['conv'],  # Matrix/conv operation. Can be {conv/fc/resnet}.
	        'weights': [64],  # Number of weights in layer.
	        'names': ['conv1_1'],  # Name of the layer.
	        'filter_size': [9],  # Filter size in the case of conv/resnet layer.
	        'normalization': ['contextual'],  # Type of normalization to use. See `models/layers/normalizations.py` for details.
	        'normalization_target': ['post'], # Normalization pre- or post-conv/matrix operation.
	        'activation': ['relu'],  # Type of activation to use. See `models/layers/activations.py` for details.
	        'activation_target': ['post'], # Activation decay pre- or post-conv/matrix operation.
	        'wd_type': [None],  # Type of weight decay to use.  See `models/layers/regularizations.py` for details.
	        'wd_target': ['pre'],  # Weight decay pre- or post-conv/matrix operation.
		}
	]
```

```
	fc_tower = [
		{  # Note that each attribute is in a list. This is because for the case of resnet you must specify multiple attributes per layer (see below for example).
	        'layers': ['fc'],  # Matrix/conv operation. Can be {conv/fc/resnet}.
	        'weights': [64],  # Number of weights in layer.
	        'names': ['fc_1'],  # Name of the layer.
	        'filter_size': [9],  # Filter size in the case of conv/resnet layer.
	        'normalization': ['contextual'],  # Type of normalization to use. See `models/layers/normalizations.py` for details.
	        'normalization_target': ['post'], # Normalization pre- or post-conv/matrix operation.
	        'activation': ['relu'],  # Type of activation to use. See `models/layers/activations.py` for details.
	        'activation_target': ['post'], # Activation decay pre- or post-conv/matrix operation.
	        'wd_type': [None],  # Type of weight decay to use.  See `models/layers/regularizations.py` for details.
	        'wd_target': ['pre'],  # Weight decay pre- or post-conv/matrix operation.
		}
	]
```

```
	resnet_tower = [
		{  # Note that each attribute is in a list. This is because for the case of resnet you must specify multiple attributes per layer (see below for example).
	        'layers': ['res'],  # Matrix/conv operation. Can be {conv/fc/resnet}.
	        'weights': [[64, 64]],  # Number of weights in layer.
	        'names': ['resnet_1'],  # Name of the layer.
	        'filter_size': [9],  # Filter size in the case of conv/resnet layer.
	        'normalization': ['contextual'],  # Type of normalization to use. See `models/layers/normalizations.py` for details.
	        'normalization_target': ['post'], # Normalization pre- or post-conv/matrix operation.
	        'activation': ['relu'],  # Type of activation to use. See `models/layers/activations.py` for details.
	        'activation_target': ['post'], # Activation decay pre- or post-conv/matrix operation.
	        'wd_type': [None],  # Type of weight decay to use.  See `models/layers/regularizations.py` for details.
	        'wd_target': ['pre'],  # Weight decay pre- or post-conv/matrix operation.
		}
	]
```

