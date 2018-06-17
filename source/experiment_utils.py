import json
import os
import time
import zipfile

from pathlib import Path
from shutil import copyfile

class Experiment:

    def __init__(self, experiment_name, experiment_description,
                 framework_name, framework_version,
                 runtime_name, runtime_version,
                 studio_utils, project_utils):

        self.experiment_name = experiment_name
        self.framework_name = framework_name
        self.framework_version = framework_version
        self.runtime_name = runtime_name
        self.runtime_version = runtime_version

        self.studio_utils = studio_utils
        self.project_utils = project_utils

        self.experiment_guid = None
        self.experiment_run_guid = None

        self.training_runs = []
        self.training_references = []

        cos_credentials = studio_utils.get_cos_credentials()
        self.wml_client = studio_utils.get_wml_client()

        self.experiment_metadata = {
                    self.wml_client.repository.ExperimentMetaNames.NAME: self.experiment_name,
                    self.wml_client.repository.ExperimentMetaNames.DESCRIPTION: experiment_description,
                    self.wml_client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: {
                                    "connection": {
                                        "endpoint_url": cos_credentials["cos_service_endpoint"],
                                        "access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
                                        "secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
                                    },
                                    "source": {
                                        "bucket": project_utils.get_data_bucket(),
                                    },
                                    "type": "s3"
                                },
                    self.wml_client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: {
                                    "connection": {
                                        "endpoint_url": cos_credentials["cos_service_endpoint"],
                                        "access_key_id": cos_credentials['cos_hmac_keys']['access_key_id'],
                                        "secret_access_key": cos_credentials['cos_hmac_keys']['secret_access_key']
                                    },
                                    "target": {
                                        "bucket": project_utils.get_results_bucket(),
                                    },
                                    "type": "s3"
                                }
                }

        # If project_id is not provided, the experiment will not show up in Watson Studio's UI
        if self.project_utils.get_project_id() is not None and len(self.project_utils.get_project_id()) > 0:
            self.experiment_metadata[self.wml_client.repository.ExperimentMetaNames.TAGS] = [
                {
                    "value": "dsx-project.{}".format(self.project_utils.get_project_id()),
                    "description": "DSX project guid"
                }
            ]

    def execute(self):
        print("Starting experiment: {}".format(self.experiment_name))

        # add stored runs to experiment
        self.experiment_metadata[self.wml_client.repository.ExperimentMetaNames.TRAINING_REFERENCES] = self.training_references

        # Store new experiment in Watson Machine Learning repository
        experiment_details = self.wml_client.repository.store_experiment(meta_props=self.experiment_metadata)
        self.experiment_guid = self.wml_client.repository.get_experiment_uid(experiment_details)
        experiment_run_details = self.wml_client.experiments.run(self.experiment_guid)

        self.experiment_run_guid = experiment_run_details["metadata"]["guid"]
        self.__update_training_run_ids()

        print("Experiment started with {} training runs".format(len(self.training_references)))

        return experiment_run_details, self.experiment_guid

    def get_training_runs(self):
        return self.training_runs

    def print_experiment_summary(self):

        # Use json to pretty print a summary
        summary = {
                    "experiment_run_guid" : self.experiment_run_guid,
                    "experiment_guid": self.experiment_guid
                  }
        summary["training_runs"] = []

        for run in self.training_runs:

            # Get latest statuses for all runs
            training_run_details = self.wml_client.training.get_details(run_uid=run.get_guid())
            #print("training_run_details",json.dumps(training_run_details, indent=2))

            summary["training_runs"].append({
                "name" : run.get_name(),
                "guid" : run.get_guid(),
                "hyperparameters" : run.get_hyperparameters(),
            })
        print("\n**** Experiment Summary Start ****\n%s" % json.dumps(summary, indent=2))
        print("**** Experiment Summary End ****\n\n")

    # Write hyperparameters to a "config.json" added to the training run's experiment.zip.
    def save_hyperparameters_config(self, hyperparameters, experiment_zip):

        # "config.json" is also the file passed to our Experiments if you use Watson Studio's HPO.
        hyperparameters_file = "config.json"
        if Path(hyperparameters_file).is_file():
            os.remove(Path(hyperparameters_file))

        with open(hyperparameters_file, "w") as file:
            file.write(json.dumps(hyperparameters))

        # Append our new hyperparameters file to the existing Experiment's .zip.
        new_experiment_zip = "hpo_experiment_temp.zip"
        if Path(new_experiment_zip).is_file():
            os.remove(Path(new_experiment_zip))

        copyfile(experiment_zip,new_experiment_zip)
        z = zipfile.ZipFile(new_experiment_zip, "a")
        z.write(hyperparameters_file)

        # Remove temp file
        try:
            os.remove(hyperparameters_file)
        except OSError as err:
            print("Error deleting {}: {}".format(hyperparameters_file, err))

        return new_experiment_zip

    def add_hpo_run(self, run_name, hpo_config, command, experiment_zip, gpu_type):
        self.__add_run(run_name, hpo_config, None, command, experiment_zip, gpu_type)

    def add_training_run(self, run_name, hyperparameters, command, experiment_zip, gpu_type):
        self.__add_run(run_name, None, hyperparameters, command, experiment_zip, gpu_type)

    # Normally either hpo_config or hyperparameters will be passed but not both.
    def __add_run(self, run_name, hpo_config, hyperparameters, command, experiment_zip, gpu_type):

        if self.experiment_metadata is None:
            raise ValueError("Experiment must first be initialized")

        # Store training run for execution as part of your experiment.
        metadata = {
            self.wml_client.repository.DefinitionMetaNames.NAME: run_name,
            self.wml_client.repository.DefinitionMetaNames.FRAMEWORK_NAME: self.framework_name,
            self.wml_client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: self.framework_version,
            self.wml_client.repository.DefinitionMetaNames.RUNTIME_NAME: self.runtime_name,
            self.wml_client.repository.DefinitionMetaNames.RUNTIME_VERSION: self.runtime_version,
            self.wml_client.repository.DefinitionMetaNames.EXECUTION_COMMAND: command
        }
        definition_details = self.wml_client.repository.store_definition(experiment_zip, metadata)
        training_run_url = self.wml_client.repository.get_definition_url(definition_details)

        training_reference = {
                    "name": run_name,
                    "training_definition_url": training_run_url,
                        "compute_configuration": {"name": gpu_type}
                }

        if hpo_config is not None:
            training_reference["hyper_parameters_optimization"] = hpo_config
        self.training_references.append(training_reference)

        print("Training run %d added to experiment" % (len(self.training_references)))

        run = TrainingRun(run_name, hyperparameters, self.studio_utils, self.wml_client, self.project_utils.get_results_bucket())

        self.training_runs.append(run)

    def __update_training_run_ids(self):

        # This method waits a max of 60 seconds for the trainings runs to start
        start = time.time()

        # Populate training runs with their guid so we can look up details as needed.
        print("Extracting guids for training runs")
        runs_started = 0
        while runs_started < len(self.training_runs) and time.time() - start < 60:

            # Pause 1 second to give time for all runs to have started
            time.sleep(1)
            experiment_run_details = self.wml_client.experiments.get_run_details(self.experiment_run_guid)

            # Loop through runs to assign variables.
            for run_status in experiment_run_details["entity"]["training_statuses"]:
                for run in self.training_runs:
                    if run.get_name() == run_status["training_reference_name"] and run_status["training_guid"] is not None:
                        run.set_guid(run_status["training_guid"])
                        runs_started += 1

        print("\nexperiment_details", json.dumps(experiment_run_details, indent=2))
        if runs_started < len(self.training_runs):
            print("Unable to obtain all training run guids")
        else:
            print("All training run guids found")

# Helper class to track run details so we can match to guids and stats coming
# back from Studio
class TrainingRun:

    def __init__(self, name, hyperparameters, studio_utils, wml_client, results_bucket):

        self.studio_utils = studio_utils
        self.wml_client = wml_client
        self.name = name
        self.hyperparameters = hyperparameters
        self.results_bucket = results_bucket
        self.guid = None

    def get_name(self):
        return self.name

    def get_hyperparameters(self):
        return self.hyperparameters

    def hyperparameters_to_json(self):
        return json.dumps(self.hyperparameters, indent=4, sort_keys=True)

    def set_guid(self, guid):
        self.guid = guid

    def get_guid(self):
        return self.guid
