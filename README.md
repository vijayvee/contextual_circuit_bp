
TODO

1) Training script. Don't forget to tf.add_n([lambda * v for k, v in model.regularizations.iteritems()]).
2) Evaluation script: Visualizations w/ LRP + plotting loss curves + plotting accuracy curves across multiple models.
3) DEBUG
4) Fix the contextual model implementation. A) Think about optimal bio CRF/eCRF sizes. B) Add the ability to automatically learn these from the data


## Preparing and executing experiments

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

6. Populate database with experiments.
	a. Running with the example laid out in (3), populate your postgres DB with all possible experiment combinations: `python prepare_experiments.py --initialize --experiment_name=one_layer_conv_mlp`.
	b. If you don't want to delete you entire DB every time, omit the `--initialize` flag.
	c. I manually access the db with `psql contextual_DCN -h 127.0.0.1 -d contextual_DCN`.

7. Run models.


Main execution script
Training script
data loader
Evaluation