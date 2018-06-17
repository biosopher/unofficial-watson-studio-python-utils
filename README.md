<!--- [instructions: quick start](#Quick-Start)

[instructions: detailed](#Detailed-Setup-Instructions)-->

### Simplified batch experimentation with Watson Studio
Executing batch experiments in Watson Studio was designed to be flexibile and to fit into a range of pre-existing workflows. However this flexibility can be daunting for new users.  Now, executing experiments is as simple as:

```
experiment = Experiment("My experiment", "Test hyperparameter range",
                        "tensorflow", "1.5", "python","3.5",
                        studio_utils, project_utils)
                        
experiment.add_training_run("cnn tests", hyperparams, "python3 experiment.py", "experiment.zip", "v100x2")
experiment.execute()
```

This repository's utility classes (below) hide the underlying complexity while providing the source code so you can dig deeper as you advance.  

<p align="center">
  <img width=500 src="media/utils_explained.png?">
</p>

To help you get started, [sample experiments are available for standard use cases](../../wiki/Execute-example-batch-experiments).  These samples are designed for quick execution from the command prompt but can be easily inserted into a notebook if desired.

### Quick Start
The next section provides detailed setup instructions to complete these steps. (a) clone this repository then (b) [install the Watson Machine Learning (WML) python client](https://wml-api-pyclient-dev.mybluemix.net/).  (c)  copy credentials to [wml_credentials.json](settings/wml_credentials.json) and [cos_credentials.json](settings/cos_credentials.json) then (d) [execute the sample batch experiments]().

<p align="center">
  <img src="media/getting_started.png?">
</p>

### Detailed Setup Instructions
If you are new to Watson Studio or simply want more details on confguring the credentials files, then follow these steps:

1. Clone this repository using the "Clone or Download" button above
2. [Setup your IBM Cloud developer utilities](../../wiki/Setup-your-IBM-Cloud-developer-tools)
3. Create Watson Machine Learning (WML) services + credentials
   - Option A: [Using Watson Studio's UI](../../wiki/Create-WML-service-via-ui)
   - Option B: [Using IBM Cloud CLI](../../wiki/Create-WML-service-via-CLI)
4. [Install WML's tooling](../../wiki/Install-WML's-tooling)
5. Create Cloud Object Storage (COS) service + credentials
   - Option A: [Using Watson Studio's UI](../../wiki/Create-COS-service-via-ui)
   - Option B: [Using IBM Cloud CLI](../../wiki/Create-WML-service-via-CLI)
6. [Install COS's tooling](../../wiki/Install-COS-Python-library-and-CLI)
7. [Create a project in Watson Studio and save the project id](../../wiki/Create-new-project-then-save-the-project-id)
8. [Execute example batch experiments](../../wiki/Execute-example-batch-experiments)

### Related and Very Useful Topics
- [View results of your experiments](../../wiki/View-results-of-your-experiments)
- [How to access TensorBoard?](../../wiki/How-to-access-TensorBoard)
- [Useful COS CLI commands](../../wiki/Useful-COS-CLI-commands)
