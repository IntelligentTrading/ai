from collections import namedtuple




######### List of coins to pile up and form dataset
TRAIN_COINS_LIST_BASIC = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0)
]

TRAIN_COINS_LIST_TOP20 = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0),
    ('BCH', 0), ('EOS', 0), ('XLM', 0), ('ADA', 0), ('MIOTA', 0), ('TRX',0),
    ('NEO', 0), ('XEM', 0), ('ZEC', 0)
]

TRAIN_COINS_LIST_BOTTOM15 = []




#########  Parameters to process raw time series
DatasetType = namedtuple('DatasetType', 'res_period win_size stride future label_func num_classes  return_target')

DATASET_TRANSFORM = {
    'basic_10m_288_24_3class_return0.01': DatasetType(res_period='10min', win_size=288, stride=1, future=24, label_func='label_3class_return_target', num_classes=3, return_target=0.01),
    'short_60m_96_4_3class_return_0.02': DatasetType(res_period='60min', win_size=96, stride=1, future=4, label_func='label_3class_return_target', num_classes=3, return_target=0.02),
    'short_60m_96_8_3class_return_0.04': DatasetType(res_period='60min', win_size=96, stride=1, future=8, label_func='label_3class_return_target', num_classes=3, return_target=0.04),

}

