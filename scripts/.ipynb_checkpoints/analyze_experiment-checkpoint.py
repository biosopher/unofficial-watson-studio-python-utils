import json
import sys

# Add our source directory to the path as Python doesn't like sub-directories
sys.path.insert(0, './source')
from watson_studio_utils import WatsonStudioUtils

studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

if len(sys.argv) < 2:
    raise ValueError("An experiment run guid must be passed as the first argument")

# Download details about this experiment run
experiment_run_guid = sys.argv[1]
experiment_run_details = studio_utils.get_wml_client().experiments.get_run_details(experiment_run_guid)
print("\nExperiment details", json.dumps(experiment_run_details, indent=2))
