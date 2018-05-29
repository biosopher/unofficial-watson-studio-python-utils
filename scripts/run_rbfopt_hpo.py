import sys
import os

sys.path.insert(0, './source')
from experiment_utils import Experiment
from hpo_utils import HPOUtils
from project_utils import ProjectUtils
from watson_studio_utils import WatsonStudioUtils

# Configure an RBFOpt experiment
def get_rbfopt_search():

    search = HPOUtils()
    search.add_static_var("batch_size", 128)

    # Fashion MNIST converges around 25 epochs and CIFAR converges after 100 epochs
    search.add_static_var("epochs", 10)

    #search.add_list("optimizer", ["sgd", "adam"])

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

    return search


# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Initialize our experiment
experiment = Experiment("Fashion MNIST-RBFOpt HPO",
                        "Perform RBFOpt HPO",
                        "tensorflow",
                        "1.5",
                        "python",
                        "3.5",
                        studio_utils,
                        project_utils)

# Create random parameters to search then create a training run for each
run_count = 25
rbfopt_search = get_rbfopt_search()

# Note: "accuracy" is the objective variable being provided to RBFOpt.  For RBFOpt to function properly
# then, we must also store the "accuracy" values to "val_dict_list.json" as shown at the end of the experiment.py file
# in the .zip experiment.
#
# To use a different objective metric, pass the object name here plus update the
# experiment.py file to pass the correct values and corresponding name to the "val_dict_list.json"
#
# Likewise if you want to use iterations instead of epochs, then you must also change that value both here as well
# as in the "val_dict_list.json".
experiment_zip = os.path.join("zips", "fashion_mnist_rbfopt.zip")
rbfopt_experiment_zip = rbfopt_search.save_rbfopt_hpo(experiment_zip,
                                                      run_count,
                                                      "accuracy",
                                                      HPOUtils.TIME_INTERVAL_EPOCH,
                                                      HPOUtils.GOAL_MAXIMIZE)

# Note: We don't pass hyperparameters for this run as RBFOpt will determine the hyperparameters to pass for
# each training run as it intelligently explores the hyperparameter space for us.
experiment.add_training_run("RBFOpt search", None, "python3 experiment.py", rbfopt_experiment_zip, "k80")

# Execute experiment
experiment.execute()

# Print the current status of the Experiment.
experiment.print_experiment_summary()

# Now you'll want to continuously monitor your experiment.  To do that from the command line,
# you should use the WML CLI: bx ml monitor training-runs TRAINING_RUN_ID
