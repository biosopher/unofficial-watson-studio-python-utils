<!--- [instructions: quick start](#Quick-Start)

[instructions: detailed](#Detailed-Setup-Instructions)-->

### Simplified batch experimentation with Watson Studio
Executing batch experiments in Watson Studio was designed to be flexibile and to fit into a range of pre-existing workflows. However this flexibility can be daunting for new users.  Now executing experiments can be as simple as:

```
experiment = Experiment("My experiment", "Test hyperparameter range",
                        "tensorflow", "1.5", "python","3.5",
                        studio_utils, project_utils)
                        
experiment.add_training_run("cnn tests", hyperparams, "python3 experiment.py", "experiment.zip", "v100x2")
experiment.execute()
```

Experiments plus other core functions are exposed so you can quickly start experimenting, yet the source code for everything's available so you can dig deeper as you advance.  These are the core classes that you'll work with:

<p align="center">
  <img width=500 src="media/utils_explained.png?">
</p>

To help you get started, [several sample experiments have been provided to show you how to use these utilities]().  These scripts are designed for quick execution from the command prompt but could be easily inserted into a notebook if desired.

### Quick Start
If you want more detailed setup instructions, then see the next section, otherwise it's a simple 4-step process. (1) clone this repository then (2) [install the Watson Machine Learning (WML) python client](https://wml-api-pyclient-dev.mybluemix.net/).  (3) if you're already familiar with WML and Cloud Object Storage then you simply copy your credentials to [wml_credentials.json](settings/wml_credentials.json) and [cos_credentials.json](settings/cos_credentials.json), and (4) [execute the scripts showcasing sample batch experiments]().

<p align="center">
  <img src="media/getting_started.png?">
</p>

### Detailed Setup Instructions
If you are new to Watson Studio or simply want more details on confguring the credentials files, then follow these steps:

1. [Install IBM Cloud's developer utilities]()
2. Create WML services + credentials
   - [Using Watson Studio UI]()
   - [Using CLI]()
3. Create COS service + credentials
   - [Using Watson Studio UI]()
   - [Using CLI]()
4. [Create a new project and obtain the project id]()
5. [Execute example batch experiments]()

