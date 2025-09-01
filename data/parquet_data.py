import glob
import random
import os
import io
import json
from PIL import Image
import pyarrow.parquet as pq

from torchvision import transforms
import torch
from torch.utils.data import IterableDataset, get_worker_info

class ImageParquetDataset(IterableDataset):
    def __init__(self, data_path, item_processor):
        self.data_path = data_path
        data_paths = self.data_path.split('+') if '+' in self.data_path else [self.data_path]
        
        self.urls_all = []
        for data_path in data_paths:
            if data_path.endswith('.txt'):
                urls = [x.strip() for x in open(data_path).readlines()]
            else:
                urls = glob.glob(os.path.join(data_path, '*.parquet'))
                urls.sort()
            self.urls_all.extend(urls)
        self.urls_all.sort()

        self.item_processor = item_processor

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        if len(self.urls_all) < total_workers:
            # pad to total_workers
            self.urls_all.extend(random.choices(self.urls_all, k=total_workers-len(self.urls_all)))

        files_iter = self.urls_all[global_worker_id::total_workers]

        while True:
            random.shuffle(files_iter)

            def row_generator():
                for file_path in files_iter:
                    try:
                        parquet_file = pq.ParquetFile(file_path)
                    except: continue
                    for rg in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(rg).to_pandas()
                        for i in range(len(table)):
                            row = table.iloc[i].to_dict()
                            yield row

            for sample in row_generator():
                try:
                    yield self.item_processor.process_item(sample, group_name='ta_tok', training_mode=True)
                except Exception as e:
                    print(e)
                    continue
