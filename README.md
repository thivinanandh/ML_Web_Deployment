# MNISTCLASSIFIER for Handwritten Digits
---

This is a demo project that I created as a Teaching Assistant for the DA212-MLOps Course offered at Indian Institute of Science, Bengaluru

The Code is currently deployed on Heroku at [URL](http://ml-mnist-classifier-2.herokuapp.com/)

Classifies images based on hand written text or uploaded Image from the MNIST-Dataset


## Learning Objectives:
---

* Training/Saving the NN Model
* Modular coding style (Cookie Cutter Template)
* Configuring the use of `setup.py` to install requirements and finding sub modules inside code
* Use DVC (Data Version Control) to seperate the data control from the version control for code. 
* Use DVC workflows for autogeneration of reports of the current build
* Continious Integration (CI) using github actions
* Continious Deployment (CD) using Heroku
* Web Deployment of ML model using Flask


## Future Addition
---
* Add continious trainning of the model based on feedback from user


## Execution 
---

The project uses the cookie cutter template (https://drivendata.github.io/cookiecutter-data-science/). 

Clone the code to your local machine

### Clone the Repo

```
https://github.com/thivinanandh/ML_MINST_Deployment.git
```

Note : The best practice is to create a virtual enviroinment so that the current set of librares des not affect the global libraries. 

Create a virtul enviroiment using

```
python3 -m venv .
```

Now, Navigate inside the Project Directory `ML_MINST_Deployment` and then run  

```
source bin/activate
```
to activate the virtual enviroinment. 


<br />

Now, run the local install of the existing code using `pip`

### Initial Depedency Setup

```
pip install -e .
```
This identifies all the submodules within the code and then will install all the library requirements for the given code. 


Now, to run the flask server , run 

### Deployment using flask

```
python3 src/Webdeployment/app.py
```

Then the website can be accessed at http://127.0.0.1:5000 . 


Note : please make sure to check the port number when flask is deployed and use that port number after the `:` in the above URL
