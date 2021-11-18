from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
from utils import word_tokenize

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value

def read_news_bert(news_path, cfg, tokenizer, mode='train'):
    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}
    index = 1

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, body, _, _ = splited[:8]
            url=""
            news_index[doc_id] = index
            index += 1
            if 'title' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
                title = title.lower()
                title = tokenizer(title, max_length=cfg[cfg.experiment_type[cfg.current_stage]].num_words_title, \
                pad_to_max_length=True, truncation=True)
            else:
                title = []
            if 'abstract' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
                abstract = abstract.lower()
                abstract = tokenizer(abstract, max_length=cfg[cfg.experiment_type[cfg.current_stage]].num_words_abstract, \
                pad_to_max_length=True, truncation=True)
            else:
                abstract = []
            if 'body' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
                body = body.lower()
                body = tokenizer(body, max_length=cfg[cfg.experiment_type[cfg.current_stage]].num_words_body, \
                pad_to_max_length=True, truncation=True)
            else:
                body = []

            news[doc_id] = [title, abstract, body]

    if mode == 'train':
        # categories = list(set(categories))
        # category_dict = {}
        # index = 1
        # for x in categories:
        #     category_dict[x] = index
        #     index += 1
        #
        # subcategories = list(set(subcategories))
        # subcategory_dict = {}
        # index = 1
        # for x in subcategories:
        #     subcategory_dict[x] = index
        #     index += 1
        #
        # domains = list(set(domains))
        # domain_dict = {}
        # index = 1
        # for x in domains:
        #     domain_dict[x] = index
        #     index += 1
        return news, news_index
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def read_news(news_path, cfg, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    word_cnt = Counter()

    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, url, _, _ = splited[:8]
            update_dict(news_index, doc_id)

            title = title.lower()
            title = word_tokenize(title)
            update_dict(news, doc_id, [title, category, subcategory])
            if mode == 'train':
                word_cnt.update(title)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > cfg[cfg.experiment_type[cfg.current_stage]].filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_index, word_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

def get_doc_input(news, news_index, word_dict, cfg):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_title), dtype='int32')
    # news_category = np.zeros((news_num, 1), dtype='int32') if cfg[cfg.experiment_type[cfg.current_stage]].use_category else None
    # news_subcategory = np.zeros((news_num, 1), dtype='int32') if cfg[cfg.experiment_type[cfg.current_stage]].use_subcategory else None

    for key in tqdm(news):
        title, category, subcategory = news[key]
        doc_index = news_index[key]

        for word_id in range(min(cfg[cfg.experiment_type[cfg.current_stage]].num_words_title, len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]

    return news_title

def get_doc_input_bert(news, news_index, cfg):
    news_num = len(news) + 1
    if 'title' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
        news_title = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_title), dtype='int32')
        news_title_type = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_type = None
        news_title_attmask = None
    if 'abstract' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
        news_abstract = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_abstract), dtype='int32')
        news_abstract_type = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_type = None
        news_abstract_attmask = None
    if 'body' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
        news_body = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_body), dtype='int32')
        news_body_type = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, cfg[cfg.experiment_type[cfg.current_stage]].num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_type = None
        news_body_attmask = None
    global_dict = {}
    # breakpoint()
    for key in tqdm(news):
        title, abstract, body = news[key]
        doc_index = news_index[key]

        if 'title' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_type[doc_index] = title['token_type_ids']
            news_title_attmask[doc_index] = title['attention_mask']
        if 'abstract' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_type[doc_index] = abstract['token_type_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']
        if 'body' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_type[doc_index] = body['token_type_ids']
            news_body_attmask[doc_index] = body['attention_mask']
        if 'abstract' in cfg[cfg.experiment_type[cfg.current_stage]].news_attributes:
            vector = np.concatenate([title['input_ids'],title['token_type_ids'],title['attention_mask'],
                                     abstract['input_ids'],abstract['token_type_ids'],abstract['attention_mask'],
                                     body['input_ids'],body['token_type_ids'],body['attention_mask']])
        else:
            vector = np.concatenate([title['input_ids'], title['token_type_ids'], title['attention_mask']])
        global_dict[doc_index] = vector
    # return news_title, news_title_type, news_title_attmask, global_dict

    return news_title, news_title_type, news_title_attmask, \
               news_abstract, news_abstract_type, news_abstract_attmask, \
               news_body, news_body_type, news_body_attmask, global_dict

def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    embedding_matrix = np.zeros(shape=(len(word_dict) + 1, word_embedding_dim))
    have_word = []
    prev_tp =[]
    count = 0
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = try_decode(line[0])
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float_cast(x) for x in line[1:]]
                    if np.array(tp).shape[0] < 300:
                        # zeros = [0] * (300 - np.array(tp).shape[0])
                        # tp += zeros
                        tp = prev_tp[-1]
                        count += 1
                    prev_tp.append(tp)
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
    print("error word count", count)
    return embedding_matrix, have_word

def try_decode(word):
    try:
        word = word.decode()
    except:
        print("Error decoding")
        word = ''
        # breakpoint()
    return word

def float_cast(str):
    try:
        if re.match('\x00', str) is not None:
            print(str)
            # breakpoint()
        str_stripped = str.strip().strip(b'\x00')
        if str != str_stripped:
            print(str)
            # breakpoint()
        ret = float(str_stripped)
    except:
        ret = 0
    return ret

if __name__ == "__main__":
    pass
    # from repoc_content_kt.news_recommendation.deprecated.parameters import parse_cfg[cfg.experiment_type[cfg.current_stage]]
    # cfg[cfg.experiment_type[cfg.current_stage]] = parse_cfg[cfg.experiment_type[cfg.current_stage]]()
    # cfg[cfg.experiment_type[cfg.current_stage]].news_attributes = ['title', 'body', 'category', 'subcategory', 'domain']
    # news, news_index, category_dict, word_dict, domain_dict, subcategory_dict = read_news(
    #     "../MIND/train/news.tsv",
    #     cfg[cfg.experiment_type[cfg.current_stage]])
    # news_title, news_abstract, news_body, news_category, news_domain, news_subcategory = get_doc_input(
    #     news, news_index, category_dict, word_dict, domain_dict, subcategory_dict, cfg[cfg.experiment_type[cfg.current_stage]])
    #
    # print(category_dict)
    # print(news_category)