import json
import os
from pathlib import Path
from shutil import copyfile
import zipfile

class HPOUtils:

    # Accepted goal values for RBFOpt HPO
    GOAL_MAXIMIZE = "maximize"
    GOAL_MINIMIZE = "minimize"

    # Possible time interval for RBFOpt HPO
    TIME_INTERVAL_EPOCH = "epoch"
    TIME_INTERVAL_ITERATION = "iteration"
    TIME_INTERVAL_STEP = "step"

    def __init__(self):

        # setup base json structure
        self.hpo = {}
        self.root = {"hyper_parameters_optimization": self.hpo}
        self.hyperparameters = []
        self.hpo["hyper_parameters"] = self.hyperparameters

        self.params_ranges = {}
        self.params_powers = {}
        self.hpo_lists = {}

    def add_step_range(self, name, min_val, max_val, step):
        self.params_ranges[name] = [min_val, max_val, step]

    def add_power_range(self,name, min_val, max_val, power):
        self.params_powers[name] = [min_val, max_val, power]

    def add_static_var(self,name, value):
        self.__add_hyperparameter(name, value, value, "", -1)

    def add_list(self, name, value_list):
        self.hpo_lists[name] = value_list

    # Let WML's HPO capability execute an RBFOpt experiment for us.
    # NOTE: a current limitation of WML's HPO capability is that training runs are executed
    # synchronously rather than parallel.
    def save_rbfopt_hpo(self, experiment_zip, training_run_count, objective, time_interval, max_or_min):

        if max_or_min.lower() in [HPOUtils.GOAL_MAXIMIZE, HPOUtils.GOAL_MINIMIZE]:
            if time_interval.lower() in [HPOUtils.TIME_INTERVAL_EPOCH,
                                         HPOUtils.TIME_INTERVAL_ITERATION,
                                         HPOUtils.TIME_INTERVAL_STEP]:
                self.__set_method(training_run_count, objective, time_interval, max_or_min)
            else:
                raise Exception("Invalid time interval.  Must be 'epoch', 'iteration' or 'step'")
        else:
            raise Exception("Invalid method type.  Must be 'maximize' or 'minimize'")

        for name in self.params_ranges:
            min_val = self.params_ranges[name][0]
            max_val = self.params_ranges[name][1]
            step = self.params_ranges[name][2]
            self.__add_hyperparameter(name, min_val, max_val, "range", step)

        # add value for powers
        for name in self.params_powers:
            min_val = self.params_powers[name][0]
            max_val = self.params_powers[name][1]
            power = self.params_powers[name][2]
            self.__add_hyperparameter(name, min_val, max_val, "power", power)

        # add lists
        for name in self.hpo_lists:
            value_list = self.hpo_lists[name]
            self.__add_list_hyperparameter(name, value_list)

        return self.__save_to_zip(experiment_zip)

    def __set_method(self, training_run_count, objective, time_interval, max_or_min):

        parameters = [
            {
                # File to write the objective's values to so the RBFOpt HPO process can evaluate the performance
                # of the  current training run compared to prior runs and determine the next set
                # of hyperparameters to explore.
                "name": "filename",
                "string_value": "val_dict_list.json"
            },{
                # How many training runs should be executed
                "name":"num_optimizer_steps",
                "int_value": training_run_count
            }
        ]

        method = {
            "name": "rbfopt",
            "parameters": parameters
        }
        self.hpo["method"] = method

        if time_interval is not None:
            # e.g. 'epoch', 'iteration' or 'step'.  The same name must
            # be provided when saving your metrics logs at the end of training
            parameters.append({
                "name": "time_interval",
                "string_value": time_interval
            })

        if objective is not None:
            # e.g. "accuracy" or "loss".  The same name must be provided when saving your objecte's values
            # to <val_dict_list.json>  the end of training
            parameters.append({
                "name": "objective",
                "string_value": objective
            })

        if max_or_min is not None:
            # Must be 'maximize' or 'minimize'
            parameters.append({
                "name": "maximize_or_minimize",
                "string_value": max_or_min
            })

    def __add_hyperparameter(self, name, min_val, max_val, step_type, step_value):

        hyperparameter = {
          "name": name
        }

        if isinstance(min_val, int) and isinstance(max_val, int):
            value_type = "int_range"
        elif isinstance(min_val, float) and isinstance(max_val, float):
            value_type = "double_range"
        else:
            raise Exception("Only int or float/double values can be provided")

        hyperparameter[value_type] = {
            "min_value": min_val,
            "max_value": max_val
        }
        if step_type in ["power", "range"]:
            hyperparameter[value_type][step_type] = step_value

        self.hyperparameters.append(hyperparameter)

    def __add_list_hyperparameter(self, name, list_values):

        hyperparameter = {
            "name": name
        }

        if isinstance(list_values[0], int):
            value_type = "int_values"
        elif isinstance(list_values[0], float):
            value_type = "double_values"
        elif isinstance(list_values[0], str):
            value_type = "string_values"
        else:
            raise Exception("Only int, float/double and string values can be provided")

        hyperparameter[value_type] = list_values
        self.hyperparameters.append(hyperparameter)

    def __save_to_zip(self, experiment_zip):

        # Write hpo to file for use by WML
        hpo_params_fname = "hpo_params.json"
        if Path(hpo_params_fname).is_file():
            os.remove(hpo_params_fname)

        with open(hpo_params_fname, "w") as file:
            json_str = json.dumps(self.root, indent=4, sort_keys=True)
            # print("RBFOpt params being passed:\n%s" % json_str)
            file.write(json_str)

        # Create temp hpo experiment .zip instead of overwriting existing experiment.
        temp_zip_fname = "hpo_experiment_temp.zip"
        if Path(temp_zip_fname).is_file():
            os.remove(temp_zip_fname)

        # Combine into temp .zip
        copyfile(experiment_zip, temp_zip_fname)
        z = zipfile.ZipFile(temp_zip_fname, "a")
        z.write(hpo_params_fname)

        # Remove temp params file
        try:
            os.remove(hpo_params_fname)
        except OSError as err:
            print("Error deleting hpo_params.json: {0}".format(err))
        return temp_zip_fname
