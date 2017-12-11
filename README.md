
## Project overview:

- configuration
	+ `config.py`
	+ `setup.py`
	+ `db/credentials.py`
- experiments (stored in db)
	+ `experiments.py`
		- parameters
			-datasets
				+ `encode_dataset.py`
			-model structures (different types of normalizations, etc)
				+ e.g., `models/structs/one_layer_conv_mlp/`
			-hyper parameters
- training and evaluation
	+ `main.py --experiment={experiment_name}`


## Preparing and executing experiments:
0. If you do not already have postgres, follow the installation instructions below (for Ubuntu 14.04):
```
	sudo add-apt-repository ppa:mc3man/trusty-media
	sudo apt-get update
	sudo apt-get install postgresql python-psycopg2 libpq-dev postgresql-client postgresql-client-common
	sudo pip install --upgrade psycopg2 # Ubuntu14 version is outdated; we need some 2.7 features
```
1. Create a copy of `config.py.template` as `config.py` and adjust its attributes to point to directories containing your images and where you intend to save data.
	a. Each image dataset should be in its own folder in the root directory `self.data_root` contained in the config.

2. Setup your python environment and install postgres if you haven't already.
	a. For Ubuntu 14 or later, execute the command: `python setup.py install`.
        b. For Mac OS X 10.10 or later:
```
      psql postgres
      create role <database user> WITH LOGIN superuser password '<database user password>';
      alter role <database user> superuser;
      create database <database name> with owner <database user>;
      \q
```

3. Fix your credentials file.
	a. `cp db/credentials.py.template db/credentials.py` then edit the file to match your system and preferences.

4. Create datasets with `encode_dataset.py`
	a. For each dataset you want to encode, create a class in `dataset_processing/` by following the `mnist.py` or `cifar.py` templates.
	b. To create a dataset: `python encode_dataset.py --dataset=mnist`

5. Create models.
	a. Add a folder with the kind of model you are creating. Follow the template of `models/structs/two_layer_conv_mlp/`.
	b. Coordinate the models you create with `experiments.py`. This script will allow you to build workers that run through any number of hyperparameter/model/dataset combinations of your experiment. For instance, if you want to measure the performance of models containing different kinds of normalization. Each model (same architecture but different normalizations) has a structure documented in `models/structs/two_layer_conv_mlp/`. The name of each model is added to the `two_layer_conv_mlp` method in `experiments.py` in the field "model_struct". Adding multiple entries in other fields, such as "dataset", will have the workers run through those combinations.
	c. See "Model construction" below for details on model construction
	d. If using online hyperparameter optimization (e.g. GPyOpt), only experiment domain parameters should be in lists.

6. Populate database with experiments.
	a. Running with the example laid out in (3), populate your postgres DB with all possible experiment combinations: `python prepare_experiments.py --experiment=two_layer_conv_mlp --initialize`.
	b. If you don't want to delete you entire DB every time, omit the `--initialize` flag.
	c. I manually access the db with `psql contextual_DCN -h 127.0.0.1 -d contextual_DCN`.

7. Run models.
	a. Run a model with `CUDA_VISIBLE_DEVICES=0 python main.py --experiment={experiment_name}`.
	b. If you forgot what experiments you've created use `python main.py --list_experiments` to list all in the DB.
	c. Run a worker with `bash start_gpu_worker.sh`. Specify a GPU, then worker will continue until there are no more experiments to run in the DB.
	d
	. Every worker will save tensorflow model checkpoints, tensorflow summaries that can be viewed in tensorboard, update the DB with its progress, and after finishing, will produce plots of the training performance.

8. Running models on the cluster.
	a. Build your workers on the cluster with `sh docker_deploy_workers.sh`
	b. Kill your processes with `docker rm $(docker stop $(docker ps -a -q --filter ancestor=serrep3.services.brown.edu:5000/contextual_circuit_bp --format="{{.ID}}"))`

9. Dumping your database:
	```pg_dump -h 127.0.0.1 contextual_DCN > 12_5_17.sql```


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


## Hyperparameter Optimization

1. To add parameters to your experiment that you want to optimize, you must do the following (using filter_size as an example):
	- add the parameter name and parameter domain to `experiments.py` in the experiment's `exp` dict:
		```
		'filter_size': 4,
		'filter_size_domain': [3,4,5],
		```
		Note: for a continuous range, pass in a list with only two items (i.e. `'filter_size_domain': [5,7]` would indicate to try the continuous range between 5 and 7)
	- add `filter_size` and `filter_size_domain` to `db/db_schema.txt` in the experiment table:
		```
		CREATE TABLE experiments ( id bigserial primary key, . . . filter_size int, filter_size_domain json)
		```
	- add the following to the `experiment_fields` function in `db/db.py`
		```            
		'filter_size' : ['experiments', 'hp_combo_history'],
		'filter_size_domain' : ['experiments', 'hp_combo_history'],
		```
	- add `filter_size` and `filter_size_domain` to `db/db.py` in the `populate_db` function, following how the other variables are added
	- add the following to the `hp_opt_dict` function in `hp_opt_utils.py`
		```
		'filter_size_domain': 'filter_size'
		```
2. Currently, this functionality uses [GPyOpt](https://github.com/SheffieldML/GPyOpt) to do Bayesian Optimization of hyperparameters. In the future, other methods will be added.

TODO:

- Evaluation script: Visualizations w/ LRP/smoothed gradient.
- Add documentation for creating 3d conv models (should be as simple as changing the data-loader shapes)

