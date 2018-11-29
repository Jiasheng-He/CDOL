from tqdm import tqdm
PATH = '../data/'
MAX_LONG = 122.081742  # 最大经度
MIN_LONG = 120.850002  # 最小经度
MAX_LAT = 31.883328  # 最大纬度
MIN_LAT = 30.666678  # 最小纬度
TAXI_NUM = 12347

LENGTH = 40  # 经度划分的栅格数
WIDTH = 40 # 纬度划分的栅格数
TIME_WINDOW = 30  # 特征时间窗（分钟

long_list = [MIN_LONG + (MAX_LONG - MIN_LONG) * i / LENGTH for i in range(LENGTH)]  # 经度区段
lat_list = [MIN_LAT + (MAX_LAT - MIN_LAT) * i / WIDTH for i in range(WIDTH)]  # 纬度区段

carhisgeo_dic = {}
car0_list = [[] for i in range(LENGTH*WIDTH)]  # 每个区域的当前的空车数量
car1_list = [[] for i in range(LENGTH*WIDTH)]  # 每个区域的当前的载客车数量
car1r_list = [[] for i in range(LENGTH*WIDTH)]  # 每个区域在过去TIME_WINDOW分钟的接单数量


def geo_pro(long_temp, lat_temp):
    """
    turn [lng,lat] to Geo
    :param long_temp: longtitude
    :param lat_temp: latitude
    :return: Geo
    """
    wid_no = 0
    len_no = 0
    for no, i in enumerate(long_list):
        if no == LENGTH - 1:
            len_no = no
            break
        if (long_temp > i) & (long_temp < long_list[no + 1]):
            len_no = no
            break

    for no, i in enumerate(lat_list):
        if no == WIDTH - 1:
            wid_no = no
            break
        if (lat_temp > i) & (lat_temp < lat_list[no + 1]):
            wid_no = no
            break

    geo_id = wid_no * LENGTH + len_no
    return int(geo_id)


def feat_eng(line_data):
    '''
    taxi distribution features
    :param line_data: read streaming data
    :return: feature vectors
    '''
    empty_state = int(line_data[0])  # State
    tstamp_temp = int(line_data[1])  # Unix Timestamp
    long_temp = float(line_data[2])  # Lon
    lat_temp = float(line_data[3])  # Lat
    id = line_data[7]
    label = int(line_data[8])

    geo_id = geo_pro(long_temp, lat_temp)

    flag = 0
    his_geoid = -1
    his_tstamp = 0
    his_estate = 1
    try:
        his_geoid = carhisgeo_dic[id][0]
        his_tstamp = carhisgeo_dic[id][1]
        his_estate = carhisgeo_dic[id][2]
        carhisgeo_dic.pop(id)
    except Exception as e:
        flag = 1

    carhisgeo_dic.update({id: [geo_id, tstamp_temp, empty_state]})

    if flag == 0:
        if his_estate == 1:
            try:
                car0_list[his_geoid].remove(his_tstamp)
            except Exception as e:
                flag = 0
        else:
            try:
                car1_list[his_geoid].remove(his_tstamp)
            except Exception as e:
                flag = 0

    if empty_state == 1:
        car0_list[geo_id].append(tstamp_temp)
    else:
        car1_list[geo_id].append(tstamp_temp)

    filter_time = tstamp_temp - TIME_WINDOW*60

    while 1:
        if len(car0_list[geo_id]) > 0:
            temp_no = car0_list[geo_id][0]
            if temp_no < filter_time:
                car0_list[geo_id].remove(temp_no)
            else:
                break
        else:
            break
    while 1:
        if len(car1_list[geo_id]) > 0:
            temp_no = car1_list[geo_id][0]
            if temp_no < filter_time:
                car1_list[geo_id].remove(temp_no)
            else:
                break
        else:
            break
    while 1:
        if len(car1r_list[geo_id]) > 0:
            temp_no = car1r_list[geo_id][0]
            if temp_no < filter_time:
                car1r_list[geo_id].remove(temp_no)
            else:
                break
        else:
            break
    car0 = len(car0_list[geo_id])
    car1 = len(car1_list[geo_id])
    car1r = len(car1r_list[geo_id])

    if label == 1:
        car1r_list[geo_id].append(tstamp_temp)

    return [geo_id, car0, car1, car1r, his_estate]


if __name__ == '__main__':
    print('basic features ...')
    file_open = open(PATH+'tiny_1.csv', 'r')
    data_list = file_open.readlines()

    with open(PATH + 'basic_tiny_1.csv', 'w') as fw:
        fw.write(data_list[0].strip().replace(',', ' ')+' Geo Car0 Car1 Car1r Hstate\n')
        data_list.remove(data_list[0])
        for item in tqdm(data_list):
            line_data = item.strip().split(',')
            online_feat = feat_eng(line_data)
            for i in online_feat:
                line_data.append(str(i))
            temp_str = ''
            for i in line_data:
                temp_str += i + ' '
            fw.write(temp_str.strip()+'\n')

    file_open = open(PATH+'tiny_2.csv', 'r')
    data_list = file_open.readlines()
    # data_cp = data_list
    # data_list = data_cp[:int(len(data_cp)*0.8)]

    with open(PATH + 'basic_tiny_2.csv', 'w') as fw:
        fw.write(data_list[0].strip().replace(',', ' ')+' Geo Car0 Car1 Car1r Hstate\n')
        data_list.remove(data_list[0])
        for item in tqdm(data_list):
            line_data = item.strip().split(',')
            online_feat = feat_eng(line_data)
            for i in online_feat:
                line_data.append(str(i))
            temp_str = ''
            for i in line_data:
                temp_str += i + ' '
            fw.write(temp_str.strip()+'\n')

    print('Done')
