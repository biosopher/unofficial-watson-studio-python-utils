import json
import os
import os.path


class ProjectUtils:

    PROJECT_ID_KEY = "project_id"
    FASHION_MIST_ROOT_KEY = "fashion_mnist_buckets"
    FASHION_MIST_DATA_BUCKET_KEY = "data_bucket"
    FASHION_MIST_RESULTS_BUCKET_KEY = "results_bucket"

    DATA_SET_FASHION_MNIST = "fashion_mnist"

    def __init__(self, studio_utils):

        self.studio_utils = studio_utils
        self.settings_directory = "settings"
        os.makedirs(self.settings_directory, exist_ok=True)

        self.settings_file = os.path.join(self.settings_directory, "workshop.json")
        if os.path.exists(self.settings_file):
            with open(self.settings_file) as json_data:
                self.settings = json.load(json_data)
        else:
            print("No workshop settings found")
            self.settings_file = None

    def get_data_bucket(self):
        return self.settings[ProjectUtils.FASHION_MIST_ROOT_KEY][ProjectUtils.FASHION_MIST_DATA_BUCKET_KEY]

    def get_results_bucket(self):
        return self.settings[ProjectUtils.FASHION_MIST_ROOT_KEY][ProjectUtils.FASHION_MIST_RESULTS_BUCKET_KEY]

    def get_project_id(self):
        if ProjectUtils.PROJECT_ID_KEY in self.settings:
            if "xxxxx" in self.settings[ProjectUtils.PROJECT_ID_KEY].lower():
                # the placeholder is still present
                return None
            else:
                return self.settings[ProjectUtils.PROJECT_ID_KEY]
        else:
            return None

    def set_project_id(self, project_id):

        self.settings[ProjectUtils.PROJECT_ID_KEY] = project_id
        self.save_project_settings()

    def download_dataset(self, data_set_name):

        # only support one dataset for now
        if data_set_name is ProjectUtils.DATA_SET_FASHION_MNIST:

            print('\nCreating data and results buckets in COS')
            cos_utils = self.studio_utils.get_cos_utils()

            self.data_bucket = cos_utils.create_unique_bucket("fashion-mnist-data")
            self.results_bucket = cos_utils.create_unique_bucket("fashion-mnist-results")

            print('\nTransferring Fashion MNIST data to COS')

            train_data_file = "t10k-images-idx3-ubyte.gz"
            train_labels_file = "t10k-labels-idx1-ubyte.gz"
            test_data_file = "train-images-idx3-ubyte.gz"
            test_labels_file = "train-labels-idx1-ubyte.gz"
            train_data_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
            train_labels_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
            test_data_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
            test_labels_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"

            # Provide a save directory to rather than delete local downloaded files
            save_directory = os.path.join("data", "fashion_mnist")

            cos_utils.transfer_remote_file_to_bucket(train_data_url, train_data_file, data_bucket, save_directory=save_directory)
            cos_utils.transfer_remote_file_to_bucket(train_labels_url, train_labels_file, data_bucket, save_directory=save_directory)
            cos_utils.transfer_remote_file_to_bucket(test_data_url, test_data_file, data_bucket, save_directory=save_directory)
            cos_utils.transfer_remote_file_to_bucket(test_labels_url, test_labels_file, data_bucket, save_directory=save_directory)

            print('\nFashion MNIST data uploaded to %s' % data_bucket)
            print('Results directory created at %s' % self.results_bucket)

            self.settings[ProjectUtils.FASHION_MIST_ROOT_KEY] = {}
            self.settings[ProjectUtils.FASHION_MIST_ROOT_KEY][ProjectUtils.FASHION_MIST_DATA_BUCKET_KEY] = self.data_bucket
            self.settings[ProjectUtils.FASHION_MIST_ROOT_KEY][ProjectUtils.FASHION_MIST_RESULTS_BUCKET_KEY] = self.results_bucket
            self.save_project_settings()

    def save_project_settings(self):
        with open(self.settings_file, 'w') as outfile:
            json.dump(self.settings, outfile)
            print('Project settings stored to %s' % self.settings_file)
