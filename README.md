# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project, A classifier is built to predict if the customer of the bank will subscribe to a term deposit with the bank or not. We explored and compared the HyperDrive option, of finding the best hyperparameters for logistic regression using the Sklearn library, with the AutoML option which explores different types of classifaction models and hyperparameters.

We are explored 12 runs to find the best hyperparameters using HyperDrive. The best performance with the HyperDrive option is the accuracy of 91.60 %. Similarly, 12 iterations are used for the AutoML option so that two options can be compared. With AutoML the best performance results in accuracy of 91.66 % with the model: VotingEnsemble.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
- The first steps are generic, So mainly intialize the workspace and create a compute cluster for model training. Then prepare the data by creating a TabularDataset form the provided CSV file.

- The data was then cleaned, and split using the training script 'train.py'.

- The classifying algorithm used is logistic regression. It is used to estimate discrete values based on a set of independent variables.

- Next, SKLearn estimator was constructed. This estimator will provide a simple way of deploying the training job on the compute target.

- The last step is to provide the Sampling Parameter to run the hyperparameter tuning. The ranges for the inverse of the regularization strength and choices for maximum number of iterations to converge are provided.

- We configure the HyperDrive to set the 'Accuracy' as the primary metric.

- Finally sibmit the experiment and find the best model.

**What are the benefits of the parameter sampler you chose?**
- Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs and thereby reducing computation costs and speedup up the exploration of the parameter space.

**What are the benefits of the early stopping policy you chose?**
- Bandit policy is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run. Slack factor is the slack allowed with respect to the best performing training run.


## AutoML
In the AutoML option, we have used the task as "clasification" and primary metric as "accuracy" and iterations as 12, so that we can compare this option with the HyperDrive option. The best run with AutoML was given by the model: VotingEnsemble with accuracy of 91.66%.

## Pipeline comparison
In our experiments, the performance of 2 options is comparable, with HyperDrive option having an accuracy of 91.60 % while AutoML option having accuracy of 91.66 %. Even with consideration to the limitation of our experiments and limited data points, we can say that AutoML will result in a better performance as AutoML go through multiple classifciation models while the HyperDrive option just uses the Logistic Regression Algorithm.

## Future work
With HyperDrive, we can increase the max_total_runs parameter allowing us to go through more hyperparameter options. We are doing intial search using Random Sampling, we can refine and narrow our search to find the best hyperparameters and use grid sampling to do so. With AutoML, we can increase the number of iteration allowing us to go through more models supported by AutoML for classifcation. Trying more models will help us find the best model for the problem in hand.
