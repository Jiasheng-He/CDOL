import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from sklearn import metrics
PATH = '../data/advanced.libSVM'

FOCUS = 739
Row_Center = FOCUS % 40
Colum_Center = int(FOCUS / 40)



LENGTH = 40
WIDTH = 40

sig_func = lambda x: 1 / (1 + math.exp(-x)) if x > -100 else 1 / (1 + math.exp(100))


with open(PATH, 'r') as fr:
     data_whole = fr.readlines()

train_mark = int(len(data_whole)*0.8)
train_df = data_whole[:train_mark]
test_df = data_whole[train_mark:]


class FTRLModel:
    def __init__(self, gid_set, dlength):
        self.gset = gid_set
        self.lambda1 = 0.001
        self.lambda2 = 0.001
        self.beta = 1
        self.alpha = 1
        self.c = 0.001

        self.w = np.zeros(dlength)
        self.z = np.zeros(dlength)
        self.n = np.zeros(dlength)

        self.yd = np.zeros(LENGTH*WIDTH)
        self.v = np.zeros(LENGTH*WIDTH)

    def update_grad(self, line_data_temp, label, gid, flag):
        line_data_temp.pop(0)
        line_data = []
        no_line_data = []
        if NO_DISTRIBUTION:
            line_data_temp.pop(0)
            line_data_temp.pop(0)
            line_data_temp.pop(0)

        for i in line_data_temp:
            no_line_data.append(int(i.split(':')[0])- 1 )
            line_data.append(float(i.split(':')[1]))


        for entry, data_item in enumerate(line_data):
            entry = no_line_data[entry]
            if (math.fabs(self.z[entry]) <= self.lambda1) | (self.z[entry] == 0):
                self.w[entry] = 0
            else:
                if self.z[entry] > 0:
                    self.w[entry] = -1 / (
                            (self.beta + math.sqrt(self.n[entry])) / self.alpha + self.lambda2) * (
                                            self.z[entry] - self.lambda1)
                else:
                    self.w[entry] = -1 / (
                            (self.beta + math.sqrt(self.n[entry])) / self.alpha + self.lambda2) * (
                                            self.z[entry] + self.lambda1)

        p = 0
        for entry, data_item in enumerate(line_data):
            entry = no_line_data[entry]
            p += self.w[entry] * data_item

        for entry, data_item in enumerate(line_data):
            entry = no_line_data[entry]
            if data_item != 0:
                if flag == 1:
                    gi = ((1 + self.c) * sig_func(p) - label - self.c * self.yd[gid] + self.v[gid]) * data_item
                else:
                    gi = (sig_func(p) - label) * data_item
                di = (math.sqrt(self.n[entry] + pow(gi,2)) - math.sqrt(self.n[entry]))/self.alpha
                self.z[entry] += gi - di * self.w[entry]
                self.n[entry] += pow(gi, 2)

    def update_vy(self, line_data_temp, gid, other_w):
        line_data_temp.pop(0)
        if NO_DISTRIBUTION:
            line_data_temp.pop(0)
            line_data_temp.pop(0)
            line_data_temp.pop(0)
        line_data = []
        no_line_data = []
        for i in line_data_temp:
            no_line_data.append(int(i.split(':')[0])- 1)
            line_data.append(float(i.split(':')[1]))

        p = 0
        for entry, data_item in enumerate(line_data):
            entry = no_line_data[entry]
            p += self.w[entry] * data_item

        p2 = 0
        for entry, data_item in enumerate(line_data):
            entry = no_line_data[entry]
            p2 += other_w[entry] * data_item
        self.yd[gid] = (sig_func(p) + sig_func(p2))/2
        self.v[gid] += self.c * (sig_func(p) - self.yd[gid])







def loss_print(amodel_list):
    loss_sum = 0
    square_sum = 0
    final_list = []
    pred_list = []
    true_list = []
    for tno in range(len(test_df)):
        item = test_df[tno]
        tgid = int(float(item.split()[-35].split(':')[1]))
        if tgid <0: tgid = 0
        label_value = int(float(item.split()[0]))
        true_list.append(int(label_value))
        temp_list = []
        for model_no, set_item in enumerate(ASET_LIST):
            if tgid in set_item:
                temp_list.append(model_no)

        item = item.split()
        if SINGLE == 1:
            Hour_temp = int(float(item[6].split(':')[1]))
            item.append(str(Hour_temp*1600 + tgid + 8000) +':1')

        item.pop(0)
        line_data = []
        no_line_data = []
        if NO_DISTRIBUTION:
            item.pop(0)
            item.pop(0)
            item.pop(0)
        for i in item:
            no_line_data.append(int(i.split(':')[0])- 1 )
            line_data.append(float(i.split(':')[1]))

        if label_value > 0:
            if len(temp_list) == 1:
                p = 0
                for entry, data_item in enumerate(line_data):
                    entry = no_line_data[entry]
                    p += amodel_list[temp_list[0]].w[entry] * data_item
                loss_sum += -1 * math.log2(sig_func(p))
                pred_result = sig_func(p)
            else:
                temp_sum = 0
                for i in temp_list:
                    p = 0
                    for entry, data_item in enumerate(line_data):
                        entry = no_line_data[entry]
                        p += amodel_list[i].w[entry] * data_item
                    temp_sum += sig_func(p)
                loss_sum += -1 * math.log2(temp_sum/2)
                pred_result = temp_sum / 2
        else:
            if len(temp_list) == 1:
                p = 0
                for entry, data_item in enumerate(line_data):
                    entry = no_line_data[entry]
                    p += amodel_list[temp_list[0]].w[entry] * data_item
                loss_sum += -1 * math.log2(1 - sig_func(p))
                pred_result = sig_func(p)
            else:
                temp_sum = 0
                for i in temp_list:
                    p = 0
                    for entry, data_item in enumerate(line_data):
                        entry = no_line_data[entry]
                        p += amodel_list[i].w[entry] * data_item
                    temp_sum += sig_func(p)
                pred_result = temp_sum / 2
                loss_sum += -1 * math.log2(1 - temp_sum/2)
        pred_list.append(pred_result)
        if tno > len(test_df) - 50:
            final_list.append([pred_result,tgid])
    test_auc = metrics.roc_auc_score(true_list, pred_list)
    pd.DataFrame(final_list, columns=['Pro', 'Gid']).to_csv(PATH + 'Pro', sep=',', index=False)
    return [loss_sum/len(test_df),test_auc]


def train(single, overlap, distribution):

    global SINGLE
    global NO_OVERLAP
    global NO_DISTRIBUTION
    SINGLE = single
    NO_OVERLAP = overlap
    NO_DISTRIBUTION = distribution

    if SINGLE != 1:
        if NO_OVERLAP != 1:
            ASET_LIST = [[], [], [], []]
            for i in range(Colum_Center):
                for j in range(Row_Center + 4):
                    ASET_LIST[2].append(i * 40 + j)

            for i in range(Colum_Center + 2):
                for j in range(Row_Center, 40):
                    ASET_LIST[3].append(i * 40 + j)

            for i in range(Colum_Center, 40):
                for j in range(Row_Center - 1, 40):
                    ASET_LIST[1].append(i * 40 + j)

            for i in range(Colum_Center - 3, 40):
                for j in range(Row_Center):
                    ASET_LIST[0].append(i * 40 + j)
        else:
            ASET_LIST = [[], [], [], []]
            for i in range(Colum_Center):
                for j in range(Row_Center):
                    ASET_LIST[2].append(i * 40 + j)

            for i in range(Colum_Center):
                for j in range(Row_Center, 40):
                    ASET_LIST[3].append(i * 40 + j)

            for i in range(Colum_Center, 40):
                for j in range(Row_Center, 40):
                    ASET_LIST[1].append(i * 40 + j)

            for i in range(Colum_Center, 40):
                for j in range(Row_Center):
                    ASET_LIST[0].append(i * 40 + j)
    else:
        ASET_LIST = [[i for i in range(1600)]]

    # 模型初始化
    dlength = 8000
    if SINGLE == 1:
        dlength += 24 * 1600
    amodel_list = []
    for area_set in ASET_LIST:
        amodel_list.append(FTRLModel(area_set, dlength))

    cnt = 0
    for no in tqdm(range(len(train_df))):
        cnt += 1
        item = train_df[no]
        geo_id = int(float(item.split()[-35].split(':')[1]))
        label_value = float(item.split()[0])
        temp_list = []
        item = item.split()
        for model_no, set_item in enumerate(ASET_LIST):
            if geo_id in set_item:
                temp_list.append(model_no)

        if SINGLE == 1:
            Hour_temp = int(float(item[6].split(':')[1]))
            item.append(str(Hour_temp*1600 + geo_id + 8000) +':1')

        if len(temp_list) == 1:
            amodel_list[temp_list[0]].update_grad(item, label_value, geo_id, 0)  # 无交叠
        else:
            for i in temp_list:
                amodel_list[i].update_grad(item, label_value, geo_id, 1)

            amodel_list[temp_list[0]].update_vy(item, geo_id, amodel_list[temp_list[1]].w)
            amodel_list[temp_list[1]].update_vy(item, geo_id, amodel_list[temp_list[0]].w)

    if SINGLE != 1:
        percent = 0
        for noc, i in enumerate(amodel_list):
            cnt = 0
            for j in i.w:
                if np.fabs(j) > 0:
                    cnt += 1

            percent += cnt / (dlength)

        print('Sparsity ...')
        print(1 - percent / len(ASET_LIST))
    else:
        percent = 0
        for noc, i in enumerate(amodel_list):
            cnt = 0
            for no1, j in enumerate(i.w):
                if (np.fabs(j) > 0) & (no1 < 8000):
                    cnt += 1
            percent = cnt / (dlength)
        print('Sparsity ...')
        print(1 - percent)
    print('[LogLoss, AUC]')
    print(loss_print(amodel_list))
    print('DONE')
