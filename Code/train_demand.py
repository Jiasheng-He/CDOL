import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit

PATH = '../data/advanced.libSVMD'
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
        self.lambda1 = 0.00001
        self.lambda2 = 0.00001
        self.beta = 1
        self.alpha = 1
        self.c = 0.0001

        self.w = np.zeros(dlength)
        self.z = np.zeros(dlength)
        self.n = np.zeros(dlength)
        # 交互更新
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
            no_line_data.append(int(i.split(':')[0]) - 1)
            line_data.append(float(i.split(':')[1]))

        for entry1, data_item in enumerate(line_data):
            entry = no_line_data[entry1]
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
        for entry1, data_item in enumerate(line_data):
            entry = no_line_data[entry1]
            p += self.w[entry] * data_item

        for entry1, data_item in enumerate(line_data):
            entry = no_line_data[entry1]
            if data_item != 0:
                if flag == 1:
                    gi = (2*p - 2 * label + self.v[gid] + self.c * (p - self.yd[gid])) * data_item
                else:
                    gi = 2*(p - label) * data_item
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
            no_line_data.append(int(i.split(':')[0]) - 1)
            line_data.append(float(i.split(':')[1]))

        p = 0
        for entry1, data_item in enumerate(line_data):
            entry = no_line_data[entry1]
            p += self.w[entry] * data_item

        p2 = 0
        for entry1, data_item in enumerate(line_data):
            entry = no_line_data[entry1]
            p2 += other_w[entry] * data_item

        self.yd[gid] = (p+p2)/2
        self.v[gid] += self.c * (p - self.yd[gid])


def loss_print(amodel_list):
    loss_sum = 0
    label_sum = 0
    mape_sum = 0
    for tno in range(len(test_df)):
        item = test_df[tno]
        tgid = int(float(item.split()[-34].split(':')[1]))
        label_value = float(item.split()[0])
        temp_list = []
        for model_no, set_item in enumerate(ASET_LIST):
            if tgid in set_item:
                temp_list.append(model_no)

        item = item.split()
        item.pop(0)
        line_data = []
        no_line_data = []
        if NO_DISTRIBUTION:
            item.pop(0)
            item.pop(0)
            item.pop(0)
        for i in item:
            no_line_data.append(int(i.split(':')[0]) - 1)
            line_data.append(float(i.split(':')[1]))

        if len(temp_list) == 1:
            p = 0
            for entry1, data_item in enumerate(line_data):
                entry = no_line_data[entry1]
                p += amodel_list[temp_list[0]].w[entry] * data_item
            if p < 0 : p= 0
            loss_sum += np.fabs(p - label_value)
            mape_sum += np.fabs(p - label_value) / (p + label_value + 0.025)
        else:
            temp_sum = 0
            for i in temp_list:
                p = 0
                for entry1, data_item in enumerate(line_data):
                    entry = no_line_data[entry1]
                    p += amodel_list[i].w[entry] * data_item
                temp_sum += p / 2
            if temp_sum < 0: temp_sum = 0
            loss_sum += np.fabs(temp_sum - label_value)
            mape_sum += np.fabs(temp_sum - label_value) / (temp_sum + label_value + 0.025)
        label_sum += label_value
    return [loss_sum/label_sum, mape_sum*2/len(test_df)]


def train_demand(single, overlap, distribution):

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
    dlength = 8000
    if SINGLE == 1:
        dlength += 0

    amodel_list = []
    for area_set in ASET_LIST:
        amodel_list.append(FTRLModel(area_set, dlength))

    cnt = 0
    for no in tqdm(range(len(train_df))):
        cnt += 1
        item = train_df[no]
        geo_id = int(float(item.split()[-34].split(':')[1]))
        if geo_id < 0: geo_id = 0

        label_value = float(item.split()[0])
        temp_list = []
        item = item.split()

        for model_no, set_item in enumerate(ASET_LIST):
            if geo_id in set_item:
                temp_list.append(model_no)

        if len(temp_list) == 1:
            amodel_list[temp_list[0]].update_grad(item, label_value, geo_id, 0)  # 无交叠
        else:
            for i in temp_list:
                amodel_list[i].update_grad(item, label_value, geo_id, 1)
            amodel_list[temp_list[0]].update_vy(item, geo_id, amodel_list[temp_list[1]].w)
            amodel_list[temp_list[1]].update_vy(item, geo_id, amodel_list[temp_list[0]].w)

    print(loss_print(amodel_list))
    percent = 0
    for no, i in enumerate(amodel_list):
        cnt = 0
        for j in i.w:
            if np.fabs(j)>0:
                cnt += 1

        percent += cnt/(dlength)

    print(1 - percent/len(ASET_LIST))
    print('DONE!!!!')

# sess = tf.Session()
# x = tf.fill([100, 1], 3.)
# w = tf.Variable(tf.random_normal(shape=[1, 100]))
#
# math.exp()
# init = tf.initialize_all_variables()
# sess.run(init)
# ys = tf.matmul(w, x)
# gradient = tf.gradients(ys, w)
# fuck = sess.run(gradient)
# print('fuck')