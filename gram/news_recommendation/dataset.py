from torch.utils.data import IterableDataset, Dataset
import numpy as np
import random
from tqdm import tqdm

class DatasetTrain(IterableDataset):
    def __init__(self, filename, news_index, news_combined, cfg):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_combined = news_combined
        self.cfg = cfg

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        # breakpoint()
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length)
        user_feature = self.news_combined[click_docs]

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].npratio)
        sample_news = neg[:label] + pos + neg[label:]
        news_feature = self.news_combined[sample_news]

        return user_feature, log_mask, news_feature, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class DatasetTest(DatasetTrain):
    def __init__(self, filename, news_index, news_scoring, cfg):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_scoring = news_scoring
        self.cfg = cfg

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_docs = line[3].split()
        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length)
        user_feature = self.news_scoring[click_docs]

        candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        news_feature = self.news_scoring[candidate_news]

        return user_feature, log_mask, news_feature, labels

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class InteractionDataset(Dataset):
    def __init__(self, data_path, news_index, news_combined, cfg):
        self.data_path = data_path
        self.news_index = news_index
        self.news_combined = news_combined
        self.cfg = cfg
        print("Init dataset")
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                # line = line.strip().split('\t')
                self.data.append(line)

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]

            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def __getitem__(self, idx):

        line = self.data[idx]
        line = line.strip().split('\t')
        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].user_log_length)
        user_feature = self.news_combined[click_docs]

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.cfg[self.cfg.experiment_type[self.cfg.current_stage]].npratio)
        sample_news = neg[:label] + pos + neg[label:]
        news_feature = self.news_combined[sample_news]
        # print(len(user_feature), len(log_mask), len(news_feature), len(label), len(click_docs), len(sample_news), len(pos))
        return user_feature, log_mask, news_feature, label, click_docs, sample_news, pos

    def __len__(self):
        return len(self.data)

class InteractionDatasetTest(InteractionDataset):
    def __init__(self, data_path, news_index, news_combined, cfg):
        super().__init__(data_path, news_index, news_combined, cfg)
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split('\t')
                self.data.append(line)
                # breakpoint()

    def __getitem__(self, idx):
        line = self.data[idx]
        click_docs = line[3].split()
        click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs), self.cfg[
            self.cfg.experiment_type[self.cfg.current_stage]].user_log_length)
        # log_ids = self.news_combined[click_docs]
        candidate_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        labels = [int(i.split('-')[1]) for i in line[4].split()]
        candidate_news, candidate_mask = self.pad_to_fix_len(candidate_news, 300, True, -1) # 50 is a candidate length
        labels, _ = self.pad_to_fix_len(labels, 300, True, -1)
        # input_ids = self.news_combined[candidate_news]
        return np.array(click_docs), log_mask, np.array(candidate_news), np.array(labels), np.array(candidate_news), candidate_mask, self.trans_to_nindex(candidate_news)

    def __len__(self):
        return len(self.data)