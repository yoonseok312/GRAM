import os
from tqdm import tqdm
import random
import logging
import pickle
from collections import Counter

def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_training_data(train_data_dir, nGPU, npratio, seed):
    random.seed(seed)
    behaviors = []

    behavior_file_path = os.path.join(train_data_dir, 'behaviors.tsv')
    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            iid, uid, time, history, imp = line.strip().split('\t')
            impressions = [x.split('-') for x in imp.split(' ')]
            pos, neg = [], []
            for news_ID, label in impressions:
                if label == '0':
                    neg.append(news_ID)
                elif label == '1':
                    pos.append(news_ID)
            if len(pos) == 0 or len(neg) == 0:
                continue
            for pos_id in pos:
                neg_candidate = get_sample(neg, npratio)
                neg_str = ' '.join(neg_candidate)
                new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                behaviors.append(new_line)

    random.shuffle(behaviors)

    behaviors_per_file = [[] for _ in range(nGPU)]
    for i, line in enumerate(behaviors):
        behaviors_per_file[i % nGPU].append(line)

    logging.info('Writing files...')
    for i in range(nGPU):
        processed_file_path = os.path.join(train_data_dir, f'behaviors_np{npratio}_{seed}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


def prepare_testing_data(test_data_dir, nGPU):
    behaviors = [[] for _ in range(nGPU)]

    behavior_file_path = os.path.join(test_data_dir, 'behaviors.tsv')
    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            behaviors[i % nGPU].append(line)

    logging.info('Writing files...')
    for i in range(nGPU):
        processed_file_path = os.path.join(test_data_dir, f'behaviors_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors[i])

    return sum([len(x) for x in behaviors])

def prepare_id_to_count(seed):
    data = []
    with open(f'YOUR_FILE_PATH', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip().split('\t')
            data.append(line)
    logs = []
    for idx in tqdm(range(len(data))):
        line = data[idx]
        logs += [i for i in line[3].split()]  # log
        logs += [i for i in line[4].split()]  # pos
        logs += [i for i in line[5].split()]  # neg
    log_id_to_count = Counter(logs)
    log_id_to_count = dict(log_id_to_count)
    print("num trainable items", len(log_id_to_count))
    with open(f'YOUR_FILE_NAME', 'wb') as f:
        pickle.dump(log_id_to_count, f)
    print(seed, "done")

# prepare_id_to_count(1)
# prepare_id_to_count(3)
# prepare_id_to_count(4)
# prepare_id_to_count(5)