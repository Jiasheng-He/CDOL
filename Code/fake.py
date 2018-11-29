import pandas as pd


fuck = pd.read_csv('../data/tiny_day_2.csv', index_col=None, sep=',')


for i in range(15):
    fuck['POI_{}'.format(i)] = 1

fuck['Hum'] = 70
fuck['Tem'] = 8
fuck['Day_w'] = 2
fuck['Day_m'] = fuck['Day']
fuck['Wea'] = 1
fuck['Wind'] = 2
fuck['Air'] = 3
fuck['Event'] = 0
fuck['Mon'] = 4
fuck = fuck.drop(['Day'], axis=1)
fuck.to_csv('../data/tiny_2.csv', index=None,sep=',')