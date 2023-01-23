import numpy as np

data_info = {
    'fea_data': "data/featry.xlsx",  # an excel
    'year_data': [500,1500,2500],  # list or excel /Ma
    'fea_dim': 27,
}


test_opt = {
    'model': "save_model/",  # model save dir
    'savename': 'lastmodel',  # model save name
    'savedir': 'result',  # teat result save path
}

model_opt = {
            'which_model': 'full_connect_res',
            'outlast': 2,
            'full_connect_res':
            {
                    'layernum': 2,
                    'drop_ratio': 0.1,
                    'out1': 300,
                    'out2': 300,
            },
        }
