import pandas as pd
import pickle

# qid：问题的唯一编号
# words：问题的词表示形式
# chars：问题的字表示形式
with open('question.pkl', 'rb') as f_:
    question = pickle.load(f_)
with open('char_embed.pkl', 'rb') as f_:
    char_embed = pickle.load(f_)
with open('word_embed.pkl', 'rb') as f_:
    word_embed = pickle.load(f_)
    
# qid：问题的唯一编号
# cid：问题所对应的分类，同一类别下的问题是同一意思，不同类别下的问题则意思不同
# label：该问题是否属于这一类别，1属于，0不属于
knowledge = pd.read_csv('knowledge.csv')
train = pd.read_csv('train.csv')

# qid1：问题1的编号
# qid2:问题2的编号
test = pd.read_csv('test.csv')  # 打分文件