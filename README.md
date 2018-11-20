# Introduction to CDOL program
Released code for Cooperative Distributed Online Learning
> Necessary modules: numpy, pandas, sklearn, keras, tqdm, lightgbm
## basic_features.py
feature engineering for basic features

## advanced_features.py
feature engineering for advanced features

## train.py
distributed model fitting for classification problem

## train_demand.py
distributed model fitting for regression problem

## neural.py
neural model fitting for classification problem

## neural_demand.py
neural model fitting for regression problem

# How to run this program?
## MacOS/ Linux
> First you should download the whole program and go to the directory of Source_Code

> Then open the terminal and enter
>> bash run.sh

> Wait until ‘Next step ...’ appears on the screen and enter
>> python3 main.py -s S1_C

> Notice that S1 means the scenario in our paper and can be replaced with S2 - S5. The character 'C' means classification problem and can be replaced with 'R' to predict the taxi demand.

## Windows
> Double click run.bat and the following steps are similar as above.
