import json
import os
import sys

# Add our source directory to the path as Python doesn't like sub-directories
source_path = os.path.join("..", "source")
sys.path.insert(0, source_path)
from watson_studio_utils import WatsonStudioUtils
from project_utils import ProjectUtils

# Initialize various utilities that will make our lives easier
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)

# Did user pass a training run?
if len(sys.argv) < 2:
    raise ValueError("A training run guid must be passed as the first argument")

training_run_guid = sys.argv[1]

results_bucket = project_utils.get_results_bucket()
all_objects = studio_utils.get_cos_utils().get_all_objects_in_bucket(results_bucket, prefix=training_run_guid)

for object in all_objects:

    remote_file = object["Key"]
    local_file = remote_file
    local_file = os.path.join("training_runs", local_file)

    if local_file.rfind(os.sep) != len(local_file)-1:  # not a directory

        studio_utils.get_cos_utils().download_file(results_bucket,
                                                   remote_file,
                                                   local_file,
                                                   is_redownload=True)
