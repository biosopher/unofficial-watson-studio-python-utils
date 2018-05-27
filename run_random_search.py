import sys
import os

# Add our source directory to the path as Python doesn't like sub-directories
sys.path.insert(0, './source')
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
    search.add_power_range("num_filters_3", 4, 8, 2)  # 16 32 64 128 256
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
    search.add_power_range("dense_neurons_1", 6, 11, 2)  # 64 128 256 512 1024 2048

    search_count = 5
    return search.create_random_search(search_count)

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Initialize our experiment
experiment = Experiment("Fashion MNIST-Random",
                         "Perform random grid search",
                         "tensorflow",
                         "1.5",
                         "python",
                         "3.5",
                         studio_utils,
                         project_utils)

# Create random parameters to search then create a training run for each
search = create_random_search()
experiment_zip = os.path.join("experiments", "fashion_mnist_random_search.zip")
for index, run_params in enumerate(search):

    run_name = "run_%d" % (index + 1)
    experiment.add_training_run(run_name, run_params, "python3 experiment.py", experiment_zip, "k80")

# Execute experiment
experiment.execute()

# Print the current status of the Experiment.
experiment.print_experiment_summary()

# Now you'll want to continuously monitor your experiment.  To do that from the command line,
# you should use the WML CLI: bx ml monitor training-runs TRAINING_RUN_ID
