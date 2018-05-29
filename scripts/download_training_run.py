import json
import os
import sys

# Add our source directory to the path as Python doesn't like sub-directories
sys.path.insert(0, './source')
from watson_studio_utils import WatsonStudioUtils
from project_utils import ProjectUtils

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Did user pass a training run?
if len(sys.argv) < 2:
    raise ValueError("An training run guid must be passed as the first argument")

# Print run details as may be useful for debugging
training_run_guid = sys.argv[1]
training_run_details = studio_utils.get_wml_client().training.get_details(run_uid=training_run_guid)
print("\nTraining run details", json.dumps(training_run_details, indent=2))

# Download logs for this training run
remote_path = "%s/learner-1" % training_run_guid
remote_log_name = "training-log.txt"
remote_log_file = "%s/%s" % (remote_path,remote_log_name)

storage_dir = "training_runs"
local_path = os.path.join(storage_dir, remote_path)
local_log_file = os.path.join(local_path, remote_log_name)
os.makedirs(local_path, exist_ok=True)

print("Downloading log file to: %s" % local_log_file)
studio_utils.get_cos_utils().download_file(project_utils.get_results_bucket(),
                                           remote_log_file,
                                           local_log_file,
                                           is_redownload=True)

