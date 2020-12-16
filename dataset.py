import re
import glob
import itertools
import random
import collections
import numpy as np
import pandas as pd

# 加载分类数据

def build_vocab(X):
    # 快速建立词表
    vocab = set(itertools.chain(*X))
    char2id = {c: i for i, c in enumerate(vocab, start=2)}
    return char2id

_SW = "/home/zhiwen/workspace/dataset/stopwords/stopwords.txt"
def load_stop_words(file=_SW):
    with open(file, "r") as fp:
        stopwords = fp.read().splitlines()
    return set(stopwords)

_THUCNews = "/home/zhiwen/workspace/dataset/THUCNews-title-label.txt"
def load_THUCNews_title_label(file=_THUCNews, nobrackets=True, limit=None):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    random.Random(6677).shuffle(lines)
    titles = []
    labels = []
    for line in lines[:limit]:
        title, label = line.split("\t")
        if not title:
            continue

        # 去掉括号内容
        if nobrackets:
            title = re.sub("\(.+?\)", lambda x: "", title)

        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_taotiao_news = "/home/zhiwen/workspace/dataset/classification/taotiao-news-abc.txt"
def load_taotiao_news(file=_taotiao_news):
    with open(file, encoding="utf-8") as fd:
        text = fd.read()
    lines = text.split("\n")[:-1]
    random.Random(6677).shuffle(lines)
    titles = []
    labels = []
    for line in lines:
        title, tags, label = line.rsplit("\t", 2)
        if not title:
            continue

        titles.append(title)
        labels.append(label)
    categoricals = list(set(labels))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    clabels = [categoricals[i] for i in labels]
    return titles, clabels, categoricals

_w100k = "/home/zhiwen/workspace/dataset/classification/weibo_senti_100k/weibo_senti_100k.csv"
def load_weibo_senti_100k(file=_w100k, noe=False):
    df = pd.read_csv(file)
    df = df.sample(frac=1) # shuffle
    X = df.review.to_list()
    y = df.label.to_list()
    # 去 emoji 表情，提升样本训练难度
    if noe:
        X = [re.sub("\[.+?\]", lambda x:"", s) for s in X]
    categoricals = {"负面": 0, "正面": 1}
    return X, y, categoricals

_MOODS = "/home/zhiwen/workspace/dataset/classification/simplifyweibo_4_moods.csv"
def load_simplifyweibo_4_moods(file=_MOODS):
    df = pd.read_csv(file)
    df = df.sample(frac=1) # shuffle
    X = df.review.to_list()
    y = df.label.to_list()
    categoricals = {"喜悦": 0, "愤怒": 1, "厌恶": 2, "低落": 3}
    return X, y, categoricals

_jobs = "/home/zhiwen/workspace/dataset/company-jobs/jobs.json"
def load_company_jobs(file=_jobs):
    filters = {'投融资', '移动开发', '高端技术职位', '行政', '运营', '人力资源',
               '后端开发', '市场/营销', '销售', '产品经理', '项目管理', '运维', '测试',
               '视觉设计', '编辑', '公关', '财务', '客服', '前端开发', '企业软件'}
    df = pd.read_json(file)
    df = df.sample(frac=1)
    X = []
    y = []
    for job in df["jobs"]:
        if job["type"] not in filters:
            continue
        X.append(job["desc"])
        y.append(job["type"])
    categoricals = list(set(y))
    categoricals.sort()
    categoricals = {label: i for i, label in enumerate(categoricals)}
    y = [categoricals[i] for i in y]
    return X, y, categoricals

_LCQMC = "/home/zhiwen/workspace/dataset/LCQMC/totals.txt"
def load_lcqmc(file=_LCQMC, shuffle=True):
    with open(file, encoding="utf-8") as fd:
        lines = fd.readlines()
    if shuffle:
        random.shuffle(lines)
    X1 = []
    X2 = []
    y = []
    for line in lines:
        x1, x2, label = line.strip().split("\t")
        X1.append(x1)
        X2.append(x2)
        y.append(int(label))
    categoricals = {"匹配":1, "不匹配":0}
    return X1, X2, y, categoricals

_ATEC_CCKS = "/home/zhiwen/workspace/dataset/matching/ATEC_CCKS/totals.txt"
def load_ATEC_CCKS(file=_ATEC_CCKS):
    return load_lcqmc(file)

if __name__ == "__main__":
    # for testing
    load_stop_words()
    load_THUCNews_title_label()
    load_taotiao_news()
    load_weibo_senti_100k()
    load_simplifyweibo_4_moods()
    load_company_jobs()

