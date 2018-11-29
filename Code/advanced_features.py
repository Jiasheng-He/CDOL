import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit
import lightgbm as lgb


PATH1 = '../data/basic_tiny_1.csv'  # historical data for prior knowledge
PATH2 = '../data/basic_tiny_2.csv'  # online data for prediction


def get_df():
    data_df = pd.read_table(PATH1, sep=' ', index_col=None)
    ss_model = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_idex, valid_index in ss_model.split(data_df):
        train_df = data_df.iloc[train_idex].reset_index(drop=True)
        valid_df = data_df.iloc[valid_index].reset_index(drop=True)

    data_df2 = pd.read_table(PATH2, sep=' ', index_col=None)

    with open('../data/basic_tiny_1_train.libSVM', 'w') as fw:
        for no, item in tqdm(train_df.iterrows()):
            label = item['Label']
            del item['Label']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')

    with open('../data/basic_tiny_1_valid.libSVM', 'w') as fw:
        for no, item in tqdm(valid_df.iterrows()):
            label = item['Label']
            del item['Label']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')

    with open('../data/basic_tiny_2.libSVM', 'w') as fw:
        for no, item in tqdm(data_df2.iterrows()):
            label = item['Label']
            del item['Label']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')

    with open('../data/basic_tiny_1_train.libSVMD', 'w') as fw:
        for no, item in tqdm(train_df.iterrows()):
            label = item['Car1r']
            del item['Car1r']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')

    with open('../data/basic_tiny_1_valid.libSVMD', 'w') as fw:
        for no, item in tqdm(valid_df.iterrows()):
            label = item['Car1r']
            del item['Car1r']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')

    with open('../data/basic_tiny_2.libSVMD', 'w') as fw:
        for no, item in tqdm(data_df2.iterrows()):
            label = item['Car1r']
            del item['Car1r']
            temp_list = [str(int(label))]
            for no, element in enumerate(list(item)):
                temp_list.append('{}:{}'.format(no,element))
            fw.write(' '.join(temp_list) + '\n')




def advanced_features():
    param = {
        'num_leaves': 255,
        'learning_rate': 0.02,
        'scale_pos_weight': 1,
        'num_threads': 40,
        'objective': 'binary',
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_sum_hessian_in_leaf': 100,
    }
    train_set = lgb.Dataset('../data/basic_tiny_1_train.libSVM')
    valid_set = lgb.Dataset('../data/basic_tiny_1_valid.libSVM')
    param['metric'] = 'binary_logloss'

    # data_df2 = pd.read_table(PATH2, sep=' ', index_col=None)
    # data_df2.drop('Label', axis=1)

    clf = lgb.train(param, train_set, num_boost_round=30, valid_sets=[valid_set], early_stopping_rounds=30)
    leaf_result = clf.predict('../data/basic_tiny_2.libSVM', num_iteration=clf.best_iteration, pred_leaf=True)

    with open('../data/basic_tiny_2.libSVM','r') as fr:
        final_data = fr.readlines()

    with open('../data/advanced.libSVM', 'w') as fw:
        for no, item in enumerate(tqdm(final_data)):
            temp_str = item.strip() + ' '
            for i in range(30):
                temp_str += str(leaf_result[no][i] + 37 + i*256) + ':' + '1 '
            fw.write(temp_str.strip() + '\n')

    param = {
        'num_leaves': 255,
        'learning_rate': 0.02,
        'scale_pos_weight': 1,
        'num_threads': 40,
        'objective': 'regression',
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_sum_hessian_in_leaf': 100,
    }
    train_set = lgb.Dataset('../data/basic_tiny_1_train.libSVMD')
    valid_set = lgb.Dataset('../data/basic_tiny_1_valid.libSVMD')
    param['metric'] = 'l2'

    # data_df2 = pd.read_table(PATH2, sep=' ', index_col=None)
    # data_df2.drop('Label', axis=1)

    clf = lgb.train(param, train_set, num_boost_round=30, valid_sets=[valid_set], early_stopping_rounds=30)
    leaf_result = clf.predict('../data/basic_tiny_2.libSVMD', num_iteration=clf.best_iteration, pred_leaf=True)

    with open('../data/basic_tiny_2.libSVMD', 'r') as fr:
        final_data = fr.readlines()

    with open('../data/advanced.libSVMD', 'w') as fw:
        for no, item in enumerate(tqdm(final_data)):
            temp_str = item.strip() + ' '
            for i in range(30):
                temp_str += str(leaf_result[no][i] + 37 + i * 256) + ':' + '1 '
            fw.write(temp_str.strip() + '\n')

    print('fuck')



get_df()
advanced_features()
print('Next step ...')
