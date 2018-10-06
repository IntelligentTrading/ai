from collections import namedtuple




######### List of coins to pile up and form dataset
TRAIN_COINS_LIST_BASIC = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0)
]

TRAIN_COINS_LIST_TOP20 = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0),
    ('BCH', 0), ('EOS', 0), ('XLM', 0), ('ADA', 0), ('TRX',0),
    ('NEO', 0), ('XEM', 0), ('ZEC', 0), ('BNB',0), ('VET',0)
]

TRAIN_COINS_LIST_TOP30 = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0),
    ('BCH', 0), ('EOS', 0), ('XLM', 0), ('ADA', 0), ('TRX',0),
    ('NEO', 0), ('XEM', 0), ('ZEC', 0), ('BNB',0), ('VET',0),

]


TRAIN_COINS_LIST_BOTTOM15 = []




#########  Parameters to process raw time series
DatasetType = namedtuple('DatasetType', 'res_period win_size stride future label_func num_classes  return_target threshold_1 threshold_2')

DATASET_TRANSFORM = {
    'basic_10m_288_24_3class_return0.01': DatasetType(res_period='10min', win_size=288, stride=1, future=24,
                                                      label_func='label_3class_return_target', num_classes=3, return_target=0.01, threshold_1=None, threshold_2=None),


    # 3 classes:
    'short_60m_150_8_3class_return_0.03': DatasetType(res_period='60min', win_size=150, stride=1, future=8,
                                                     label_func='label_3class_return_target', num_classes=3,
                                                     return_target=0.03, threshold_1=None, threshold_2=None),

    'short_60m_160_8_3class_return_0.03': DatasetType(res_period='60min', win_size=160, stride=1, future=8,
                                                     label_func='label_3class_return_target', num_classes=3,
                                                     return_target=0.03, threshold_1=None, threshold_2=None),




    'medium_240m_100_12_3class_return_0.08': DatasetType(res_period='240min', win_size=100, stride=1, future=12,
                                                       label_func='label_3class_return_target', num_classes=3, return_target=0.08, threshold_1=None, threshold_2=None),




    'long_1440m_28_7_class3_return_0.11': DatasetType(res_period='1440min', win_size=28, stride=1, future=7,
                                                       label_func='label_3class_return_target', num_classes=3,
                                                       return_target=0.11, threshold_1=None, threshold_2=None),




    # 2 classes
    'long_1440m_28_4_class2': DatasetType(res_period='1440min', win_size=28, stride=1, future=4,
                                                       label_func='label_2class_return_target', num_classes=2,
                                                       return_target=0.1, threshold_1=None, threshold_2=None), # return_target does not play role here



    # hit max two thesholds
    'short_60m_168_4_maxhit3cl_0.02_04': DatasetType(res_period='60min', win_size=168, stride=1, future=4,
                                                     label_func='label_3class_max_hit', num_classes=3,
                                                     return_target=None, threshold_1=0.02, threshold_2=0.04),




    # hit max one theshold
    'short_60m_168_4_maxhit2cl_0.02': DatasetType(res_period='60min', win_size=168, stride=1, future=4,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.02, threshold_2=None ),

    'short_60m_168_4_maxhit2cl_0.03': DatasetType(res_period='60min', win_size=168, stride=1, future=4,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.03, threshold_2=None ),

    'short_60m_168_4_maxhit2cl_0.04': DatasetType(res_period='60min', win_size=168, stride=1, future=4,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.04, threshold_2=None ),


    'short_60m_168_8_maxhit2cl_0.02': DatasetType(res_period='60min', win_size=168, stride=1, future=8,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.02, threshold_2=None ),

    'short_60m_168_8_maxhit2cl_0.03': DatasetType(res_period='60min', win_size=168, stride=1, future=8,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.03, threshold_2=None ),

    'short_60m_168_8_maxhit2cl_0.04': DatasetType(res_period='60min', win_size=168, stride=1, future=8,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.04, threshold_2=None ),

    'short_60m_168_8_maxhit2cl_0.05': DatasetType(res_period='60min', win_size=168, stride=1, future=8,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.05, threshold_2=None ),


    'medium_240m_100_7_maxhit2cl_0.1': DatasetType(res_period='240min', win_size=100, stride=1, future=7,
                                                     label_func='label_2class_max_hit', num_classes=2,
                                                     return_target=None, threshold_1=0.1, threshold_2=None )
}

