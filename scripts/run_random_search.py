import sys
import os

# Add our source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..","source")
sys.path.insert(0, source_path)
from random_search import RandomSearch
from watson_studio_utils import WatsonStudioUtils
from project_utils import ProjectUtils
from experiment_utils import Experiment

# Configure a random search
def create_random_search():

    search = RandomSearch()
    search.add_static_var("batch_size", 128)

    # Fashion MNIST converges around 25 epochs and CIFAR converges after 100 epochs
    search.add_static_var("epochs", 10)

    search.add_list("optimizer", ["sgd", "adam"])

    search.add_power_range("num_filters_1", 5, 8, 2)  # 32 64 128 256
    search.add_power_range("num_filters_2", 4, 8, 2)  # 16 32 64 128 256
    search.add_power_range("num_filters_3", 4, 7, 2)  # 16 32 64 128
    search.add_step_range("filter_size_1", 2, 3, 1)
    search.add_step_range("filter_size_2", 2, 3, 1)
    search.add_step_range("filter_size_3", 2, 3, 1)
    search.add_step_range("pool_size_1", 2, 2, 1)
    search.add_step_range("pool_size_2", 2, 2, 1)
    search.add_step_range("pool_size_3", 2, 2, 1)
    search.add_step_range("dropout_1", 0.1, 0.9, 0.1)
    search.add_step_range("dropout_2", 0.1, 0.9, 0.1)
    search.add_step_range("dropout_3", 0.1, 0.9, 0.1)
    search.add_step_range("dropout_4", 0.1, 0.9, 0.1)
    search.add_power_range("dense_1", 6, 11, 2)  # 64 128 256 512 1024 2048

    search_count = 5
    return search.create_random_search(search_count)

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

isPyTorch = False  # else TensorFlow
if isPyTorch:
    framework = "pytorch"
    version = "0.4"
    experiment_zip = "dynamic_hyperparms_pt.zip"
else:
    framework = "tensorflow"
    version = "1.5"
    experiment_zip = "dynamic_hyperparms_tf.zip"

# Initialize our experiment
gpu_type = "k80"
experiment = Experiment( "Fashion MNIST-Custom Random-{}-{}".format(framework, gpu_type),
                         "Perform random grid search",
                         framework,
                         version,
                         "python",
                         "3.5",
                         studio_utils,
                         project_utils)

experiment_zip = os.path.join("experiment_zips", experiment_zip)

# Create random parameters to search then create a training run for each
search = create_random_search()
for index, run_params in enumerate(search):

    # Append hyperparameters to the command
    command = "python3 experiment.py"

    # Specify different GPU types as "k80", "k80x2", "k80x4", "p100", ...
    run_name = "run_%d" % (index + 1)

    # Add the hyperparameters to the experiment.zip (in config.json)
    updated_experiment_zip = experiment.save_hyperparameters_config(run_params, experiment_zip)

    experiment.add_training_run(run_name, command, updated_experiment_zip, gpu_type)

# Execute experiment
experiment.execute()

# Print the current status of the Experiment.
#experiment.print_experiment_summary()

# Now you'll want to continuously monitor your experiment.  To do that from the command line,
# you should use the WML CLI: bx ml monitor training-runs TRAINING_RUN_ID
