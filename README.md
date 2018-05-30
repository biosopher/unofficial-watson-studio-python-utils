<!--- [instructions: quick start](#Quick-Start)

[instructions: detailed](#Detailed-Setup-Instructions)-->

### Simplified Batch training with Watson Studio
Batch training in Watson Studio was designed to be flexibile and fit into a range of pre-existing deep learning workflows. However this flexibility can be daunting for new users so these utils expose the core functions required to quickly start experiment while giving you the source code so you can dig deeper as you advance.

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

