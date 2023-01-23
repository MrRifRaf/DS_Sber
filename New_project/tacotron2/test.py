import multiprocessing as mp
import os

num_workers = mp.cpu_count()

os.path.normpath(
    r'mnt\train\Cori_Samuel\mels\secretagent_10_conrad_0187.mel'.replace(
        '\\', '/'))
