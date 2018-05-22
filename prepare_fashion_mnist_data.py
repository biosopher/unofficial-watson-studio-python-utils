import sys

sys.path.insert(0, './source')
from watson_studio_utils import WatsonStudioUtils
from project_utils import ProjectUtils

print('\nCreating data and results buckets in COS')
studio_utils = WatsonStudioUtils(region="us-south")
studio_utils.configure_utilities_from_file()

project_utils = ProjectUtils(studio_utils)
project_utils.download_dataset(ProjectUtils.DATA_SET_FASHION_MNIST)

