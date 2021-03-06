import sys
import os

# Add source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..", "source")
sys.path.insert(0, source_path)
from watson_studio_utils import WatsonStudioUtils
from experiment_utils import Experiment
from project_utils  import ProjectUtils

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Initialize our experiment
experiment = Experiment("Fashion MNIST-dropout tests",
                        "Test two different dropout values",
                        "tensorflow",
                        "1.5",
                        "python",
                        "3.5",
                        studio_utils,
                        project_utils)

# Add two training runs to determine which dropout is best: 0.4 or 0.9
run_1a_path = os.path.join("experiment_zips", "dropout_0.4.zip")
run_1b_path = os.path.join("experiment_zips", "dropout_0.6.zip")

# Specify different GPU types as "k80", "k80x2", "k80x4", "p100", ...
experiment.add_training_run("Run #1", "python3 experiment.py", run_1a_path, "k80")
experiment.add_training_run("Run #2", "python3 experiment.py", run_1b_path, "k80")

# Execute experiment
experiment.execute()

# Print the current status of the Experiment.
experiment.print_experiment_summary()

# Now you'll want to continuously monitor your experiment.  To do that from the command line,
# you should use the WML CLI.
