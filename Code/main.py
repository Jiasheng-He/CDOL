import argparse, sys
from train import *
from train_demand import *
from neural import *
from neural_demand import *

def parse_args():
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='scenario', default='S1_C', type=str)
    args = vars(parser.parse_args())
    return args

scenario_list = ['S1_C','S2_C','S3_C','S4_C','S5_C','S1_R','S2_R','S3_R','S4_R','S5_R']

if __name__ == '__main__':
    args = parse_args()
    scenario = args['scenario']
    if scenario in scenario_list:
        if scenario.split('_')[1] == 'C':
            if scenario.split('_')[0][1] == '1':
                train(0,0,0)
            if scenario.split('_')[0][1] == '2':
                train(0,1,0)
            if scenario.split('_')[0][1] == '3':
                train(1,0,0)
            if scenario.split('_')[0][1] == '4':
                train(0,0,1)
            if scenario.split('_')[0][1] == '5':
                neural()
        else:
            if scenario.split('_')[0][1] == '1':
                train_demand(0, 0, 0)
            if scenario.split('_')[0][1] == '2':
                train_demand(0, 1, 0)
            if scenario.split('_')[0][1] == '3':
                train_demand(1, 0, 0)
            if scenario.split('_')[0][1] == '4':
                train_demand(0, 0, 1)
            if scenario.split('_')[0][1] == '5':
                neural_demand()

    else:
        print('ERROR')