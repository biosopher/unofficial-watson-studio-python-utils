import json
import os
import sys

# Add our source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..","source")
sys.path.insert(0, source_path)
from watson_studio_utils import WatsonStudioUtils
from project_utils import ProjectUtils

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Did user pass a training run?
if len(sys.argv) < 2:
    raise ValueError("An experiment run guid must be passed as the first argument")

# Print run details as may be useful for debugging
experiment_run_guid = sys.argv[1]
experiment_run_details = studio_utils.get_wml_client().experiments.get_run_details(experiment_run_guid)
print("\nExperiment run details", json.dumps(experiment_run_details, sort_keys=True, indent=4))

training_runs = experiment_run_details["entity"]["training_statuses"]
for run in training_runs:
    run_guid = run["training_guid"]

    # Download logs for this training run
    remote_path = "%s/learner-1" % run_guid
    remote_log_name = "training-log.txt"
    remote_log_file = "%s/%s" % (remote_path,remote_log_name)

    local_path = os.path.join("experiment_runs", experiment_run_guid)
    os.makedirs(local_path, exist_ok=True)

    local_log_file = os.path.join(local_path, "%s-log.txt" % run_guid)
    print("Downloading log file to: %s" % local_log_file)
    studio_utils.get_cos_utils().download_file(project_utils.get_results_bucket(),
                                               remote_log_file,
                                               local_log_file,
                                               is_redownload = True)
