# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project, A classifier is built to predict if the customer of the bank will subscribe to a term deposit with the bank or not. HyperDrive option is explored to find the best hyperparameters for logistic regression using the Sklearn library, and compared with AutoML option which explores different types of classifaction models and hyperparameters.

12 runs are explored to find the best hyperparameters using HyperDrive. The best performance with the HyperDrive option is the accuracy of 91.47%. Similarly, 12 iterations are used for the AutoML option so that two options can be compared. With AutoML the best performance results in accuracy of 91.52 % with the model: XGBoostClassifier.

## Scikit-learn Pipeline
**Pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
- Intialized the workspace and created a compute cluster for model training. Then preparde the data by creating a TabularDataset from provided CSV file.
- The data is cleaned, and split using the training script 'train.py'.
- Logistic Regression algorithm is used for Classification. It is used to estimate discrete values based on a set of independent variables.
- SKLearn estimator was constructed. This estimator will provide a simple way of deploying the training job on the compute target.
- Sampling Parameter is provided to run the hyperparameter tuning.
- HyperDrive is configured to set the 'accuracy' as the primary metric.
- Submited the experiment and find the best model.

**Benefits of chosen parameter sampler**
- Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs and thereby reducing computation costs and speed up the exploration of the parameter space.

**Benefits of chosen early stopping policy**
- Bandit policy is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run. Slack factor is the slack allowed with respect to the best performing training run.

## AutoML
In the AutoML option, we have used the task as "clasification" and primary metric as "accuracy" and iterations as 12, so that we can compare this option with the HyperDrive option. The best run with AutoML was given by the model: XGBoostClassifier with accuracy of 91.52%. Please find below all the tried models and their performance by AutoML.
```
ITERATION   PIPELINE                                       DURATION      METRIC      BEST
         0   MaxAbsScaler LightGBM                          0:00:52       0.9151    0.9151
         1   MaxAbsScaler XGBoostClassifier                 0:01:14       0.9152    0.9152
         2   MinMaxScaler RandomForest                      0:00:58       0.8988    0.9152
         3   MinMaxScaler RandomForest                      0:01:19       0.8880    0.9152
         4   MinMaxScaler RandomForest                      0:00:51       0.8133    0.9152
         5   MinMaxScaler SVM                               0:25:17       0.9014    0.9152
         6    VotingEnsemble                                0:01:44       0.9150    0.9152
         7    StackEnsemble                                 0:01:52       0.9137    0.9152
```
## Pipeline comparison
- In our experiments, the performance of 2 options is comparable, with HyperDrive option having an accuracy of 91.47% while AutoML option having accuracy of 91.52%. Even with consideration to the limitation of our experiments and limited data points, we can say that AutoML will result in a better performance as AutoML go through multiple classifciation models while the HyperDrive option just uses the Logistic Regression Algorithm.
- In Hyperdrive we have to build a training script, but in AutoML we just need to pass the data, and define the task.
- Hyperdrive option took around 25 minutes to complete whereas AutoML took about 1 hour to complete it's execution.
- 

## Future work
- Data is highly imbalanced, so strategies to deal with class imbalance like oversampling, undersampling, SMOTE etc can be explored. We can also try grid sampling.
- With AutoML, number of iteration can be increased, allowing us to go through more models supported by AutoML for classifcation. Trying more models will help us find the best model.
- With HyperDrive, max_total_runs parameter can be increased.
