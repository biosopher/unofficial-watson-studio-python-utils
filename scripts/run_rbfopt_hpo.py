import sys
import os

source_path = os.path.join("..","source")
sys.path.insert(0, source_path)
from experiment_utils import Experiment
from hpo_utils import HPOUtils
from project_utils import ProjectUtils
from watson_studio_utils import WatsonStudioUtils

# Configure an RBFOpt experiment
def get_rbfopt_config():

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
    search.add_power_range("dense_1", 6, 11, 2)  # 64 128 256 512 1024 2048

    # Note: "accuracy" is the objective variable being provided to RBFOpt.
    # For RBFOpt to function properly then, we must also store the "accuracy"
    # values to "val_dict_list.json" as shown at the end of the experiment.py file
    # in the .zip experiment.
    #
    # To use a different objective metric, pass the object name here plus update the
    # experiment.py file to pass the correct values and corresponding name to the "val_dict_list.json"
    #
    # Likewise if you want to use iterations instead of epochs, then you must also
    # change that value both here as well as in the "val_dict_list.json".
    run_count = 10
    return search.get_hpo_config(run_count,
                                 HPOUtils.OBJECTIVE_ACCURACY,
                                 HPOUtils.TIME_INTERVAL_EPOCH,
                                 HPOUtils.GOAL_MAXIMIZE)


# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

isPyTorch = True # else TensorFlow
if isPyTorch:
    framework = "pytorch"
    version = "0.4"
    experiment_zip = "dynamic_hyperparms_pt.zip"
else:
    framework = "tensorflow"
    version = "1.5"
    experiment_zip = "dynamic_hyperparms_tf.zip"

experiment_zip = os.path.join("experiment_zips", experiment_zip)

# Initialize our experiment
gpu_type = "k80"
experiment = Experiment("Fashion MNIST-RBFOpt HPO-{}-{}".format(framework, gpu_type),
                        "Perform RBFOpt HPO",
                         framework,
                         version,
                        "python",
                        "3.5",
                        studio_utils,
                        project_utils)

# Run RBFOpt to search through the hyperparameters
hpo_config = get_rbfopt_config()

# Specify different GPU types as "k80", "k80x2", "k80x4", "p100", ...
experiment.add_hpo_run("RBFOpt search", hpo_config, "python3 experiment.py", experiment_zip, gpu_type)

# Execute experiment
experiment.execute()

# Print the current status of the Experiment.
experiment.print_experiment_summary()

# Now you'll want to continuously monitor your experiment.  To do that from the command line,
# you should use the WML CLI: bx ml monitor training-runs TRAINING_RUN_ID
