import logging

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

file_handler = logging.FileHandler('train.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)

epoch, batch_iter, i = 3, 2, 1
items_per_sec = 3.2

logger.info(dict(step=(epoch, batch_iter, i), data={
            'val_items_per_sec': items_per_sec}))
