import json
import os
import time
from watson_machine_learning_client import WatsonMachineLearningAPIClient


class Experiment:

    def __init__(self, experiment_name, experiment_description,
                 framework_name, framework_version,
                 runtime_name, runtime_version,
                 studio_utils, project_utils):

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


        self.wml_client = WatsonMachineLearningAPIClient(studio_utils.get_wml_credentials())
        print("WML client version: %s" % self.wml_client.version)

        cos_credentials = studio_utils.get_cos_credentials()
        self.experiment_metadata = {
                    self.wml_client.repository.ExperimentMetaNames.NAME: experiment_name,
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
                    "value": "dsx-project.%s" % self.project_utils.get_project_id(),
                    "description": "DSX project guid"
                }
            ]

    def add_training_run(self, run_name, hyperparameters, command, experiment_zip, gpu_type):

        if self.experiment_metadata is None:
            raise ValueError("Experiment must first be initialized")

        # Append hyperparameters to the command
        if hyperparameters is not None:
            for name in hyperparameters:
                command = "%s --%s %s" % (command, name, str(hyperparameters[name]))

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

        self.training_references.append({
            "name": run_name,
            "training_definition_url": training_run_url,
            "compute_configuration": {"name": gpu_type}
        })
        print("Training run %d added to experiment" % (len(self.training_references)))

        run = TrainingRun(run_name, hyperparameters, self.studio_utils, self.wml_client, self.project_utils.get_results_bucket())

        self.training_runs.append(run)

    def execute(self):
        print("Starting experiment: %s" % (self.experiment_metadata[self.wml_client.repository.ExperimentMetaNames.NAME]))

        # add stored runs to experiment
        self.experiment_metadata[self.wml_client.repository.ExperimentMetaNames.TRAINING_REFERENCES] = self.training_references

        # Store new experiment in Watson Machine Learning repository
        experiment_details = self.wml_client.repository.store_experiment(meta_props=self.experiment_metadata)
        self.experiment_guid = self.wml_client.repository.get_experiment_uid(experiment_details)
        experiment_run_details = self.wml_client.experiments.run(self.experiment_guid)

        self.experiment_run_guid = experiment_run_details["metadata"]["guid"]
        self.__update_training_run_ids()

        print("Experiment started with %d training runs" % len(self.training_references))

        return experiment_run_details, self.experiment_guid

    def get_training_runs(self):
        return self.training_runs

    def print_experiment_summary(self):

        # Use json to pretty print a summary
        summary = {
                    "experiment_run_guid" : self.experiment_guid,
                    "experiment_guid": self.experiment_run_guid
                  }
        summary["training_runs"] = []

        for run in self.training_runs:

            # Get latest statuses for all runs
            run.update_status()
            summary["training_runs"].append({
                "name" : run.get_name(),
                "guid" : run.get_guid(),
                "status" : run.get_status(),
                "hyperparameters" : run.get_hyperparameters(),
            })
        print("\n**** Experiment Summary Start ****\n%s" % json.dumps(summary, indent=2))
        print("**** Experiment Summary End ****\n\n")

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
            print("\nexperiment_details", json.dumps(experiment_run_details, indent=2))

            # Loop through runs to assign variables.
            for run_status in experiment_run_details["entity"]["training_statuses"]:
                for run in self.training_runs:
                    if run.get_name() == run_status["training_reference_name"]:
                        run.set_guid(run_status["training_guid"])
                        runs_started += 1

        if runs_started < len(self.training_runs):
            print("Unable to obtain all training run guids")
        else:
            print("All training run guids found")

class TrainingRun:

    def __init__(self, name, hyperparameters, studio_utils, wml_client, results_bucket):

        self.studio_utils = studio_utils
        self.wml_client = wml_client
        self.name = name
        self.hyperparameters = hyperparameters
        self.results_bucket = results_bucket
        self.guid = None

        self.status = "pending"

        self.train_accuracy = "not found"
        self.train_loss = "not found"
        self.test_accuracy = "not found"
        self.test_loss = "not found"
        self.train_time = "not found"

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

    def download_training_log(self, working_directory):

        remote_log_file = "%s/learner-1/training-log.txt" % self.get_guid
        local_log_path = remote_log_file
        if not os.path.isfile(local_log_path):
            # Log not yet downloaded
            self.studio_utils.get_cos_utils().download_file(self.results_bucket, remote_log_file, local_log_path)

        return local_log_path

    def update_status(self):

        if self.status in ["pending"]:
            training_run_details = self.wml_client.training.get_details(run_uid=self.get_guid())
            print("training_run_details",json.dumps(training_run_details, indent=2))

    def get_status(self):
        return self.status

    def download_final_results(self, working_directory):

        # Query the server for current status
        training_run_details = self.wml_client.training.get_details(run_uid=self.guid)
        #status = training_run_details[""]
        #final_results = {"status" : status}

        print("training_run_details", training_run_details)
        #print("status for %s: " % (self.get_guid, status))

        # if status is not "pending":

        local_log_file = self.download_training_log(working_directory)
        if os.path.isfile(local_log_file):  # Log was downloaded
            with open(local_log_file, 'r') as file:
                lines = file.readlines()
                # Remove whitespace characters like `\n` at the end of each line
                lines = [x.strip() for x in lines]
                for line in lines:
                    if line.startswith("Final train accuracy:"):
                        self.train_accuracy = line.replace("Final train accuracy:", "").strip()
                    elif line.startswith("Final train loss:"):
                        self.train_loss = line.replace("Final train loss:", "").strip()
                    elif line.startswith("Final test accuracy:"):
                        self.test_accuracy = line.replace("Final test accuracy:", "").strip()
                    elif line.startswith("Final test loss:"):
                        self.test_loss = line.replace("Final test loss:", "").strip()
                    elif line.startswith("Total train time:"):
                        self.train_time = line.replace("Total train time:", "").strip()
                file.close()

        return final_results


