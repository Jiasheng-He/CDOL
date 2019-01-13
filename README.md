# Introduction to CDOL project
Released code for Cooperative Distributed Online Learning using Python3
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

# How to run this project?
> 1. First you should download the whole project.
## MacOS/ Linux
> 2. Then open the terminal, go to the directory 'Code' and enter
>> bash run.sh

> 3. Wait until ‘Next step ...’ appears on the screen and enter
>> python3 main.py -s S1_C

>> Notice that S1 means the corresponding scenario in our paper and can be replaced with S2 - S5. The character 'C' means classification problem and can be replaced with 'R' to predict the taxi demand.

S1 --- CDOL
S2

## Windows
> 2. Double click run.bat and the following steps are similar as above.
