# coding: utf-8
# author: lu yf
# create date: 2018/8/11

from __future__ import division
import os
import random
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, f1_score, roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
import warnings
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import numpy as np

warnings.filterwarnings('ignore')
random.seed(1)


class Evaluation:

    def __init__(self, embeddings_data):
        self.embeddings_data = embeddings_data
        self.name_emb_dict = {}

    def load_embeddings(self):
        embeddings_mat = scipy.io.loadmat(self.embeddings_data)
        key = filter(lambda k:k.startswith('_') is False,embeddings_mat.keys())[0]

        embeddings = embeddings_mat[key]
        for i in range(len(embeddings)):
            self.name_emb_dict[i] = embeddings[i]

    def kmeans_nmi(self,cluster_k):
        x = []
        y = []
        with open('../data/yelpWWW/oriData/business_category.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))
        print(len(x))
        km = KMeans(n_clusters=cluster_k)
        km.fit(x, y)
        y_pre = km.predict(x)
        # y_pre = km.fit_predict(x,y)
        nmi = normalized_mutual_info_score(y, y_pre)
        print('Kmean, k={}, nmi={}'.format(cluster_k, nmi))
        return nmi

    def classification(self,train_size):
        x = []
        y = []
        with open('../data/yelpWWW/oriData/business_category.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))

        print(len(x))
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1-train_size,random_state=9)
        # print ('train_size: {}'.format(train_sicv ze))
        lr = LogisticRegression()

        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)

        micro_f1 = f1_score(y_valid, y_valid_pred,average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred,average='macro')
        print ('Macro_F1_score:{}'.format(macro_f1))
        print ('Micro_F1_score:{}'.format(micro_f1))
        return macro_f1,micro_f1

    def calculate_sim(self,u,v,sum_flag):
        if sum_flag:
            return sum(np.abs(np.array(u)-np.array(v)))
            # return sum(np.abs(np.array(u)*np.array(v)))
        else:
            return np.abs(np.array(u)-np.array(v))

    def binary_classification_aa(self, x_train, y_train, x_test, y_test):
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, classifier.predict(x_test))
        acc = accuracy_score(y_test,classifier.predict(x_test))
        print('auc: {}'.format(auc_score))
        print('f1: {}'.format(f1))
        print('acc: {}'.format(acc))

    def pre_4_link_prediction(self,data_type):
        print('prepare {} data for link prediction...'.format(data_type))
        x = []
        y = []
        with open('../data/yelpWWW_lp/pos_brurb_'+data_type+'.txt', 'r') as p_co_file:
            for line in p_co_file:
                tokens = line.strip().split('\t')
                a1_name = int(tokens[0])
                a2_name = int(tokens[1])
                if self.name_emb_dict.has_key(a1_name) and self.name_emb_dict.has_key(a2_name):
                    a1_emb = self.name_emb_dict[a1_name]
                    a2_emb = self.name_emb_dict[a2_name]
                    sim_a1_a2 = self.calculate_sim(a1_emb, a2_emb, sum_flag=False)
                    x.append(sim_a1_a2)
                    y.append(1)
        with open('../data/yelpWWW_lp/neg_brurb_' + data_type+'.txt', 'r') as p_co_file:
            for line in p_co_file:
                tokens = line.strip().split('\t')
                a1_name = int(tokens[0])
                a2_name = int(tokens[1])
                if self.name_emb_dict.has_key(a1_name) and self.name_emb_dict.has_key(a2_name):
                    a1_emb = self.name_emb_dict[a1_name]
                    a2_emb = self.name_emb_dict[a2_name]
                    sim_a1_a2 = self.calculate_sim(a1_emb, a2_emb, sum_flag=False)
                    x.append(sim_a1_a2)
                    y.append(0)
        return x,y

    def link_prediction_with_auc(self):
        train_x, train_y = self.pre_4_link_prediction('train')
        test_x, test_y = self.pre_4_link_prediction('test')
        print(len(train_x), len(test_x))
        self.binary_classification_aa(train_x, train_y, test_x, test_y)
        # x_train, x_valid, y_train, y_valid = train_test_split(test_x, test_y, test_size=1 - 0.8, random_state=9)
        # self.binary_classification_aa(x_train, y_train, x_valid, y_valid)

    def new_cl(self):
        x_train, x_valid, y_train, y_valid = [],[],[],[]
        new_nodes = []
        with open('../yelp_delta_nodes.txt', 'r') as new_node_file:
            for f in new_node_file:
                new_nodes.append(int(f.strip()))

        with open('../data/yelpWWW/oriData/business_category.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()

        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                if int(tokens[0]) in new_nodes:
                    x_valid.append(list(self.name_emb_dict[int(tokens[0])]))
                    y_valid.append(int(tokens[1]))
                else:
                    x_train.append(list(self.name_emb_dict[int(tokens[0])]))
                    y_train.append(int(tokens[1]))

        # print (len(x_train),len(x_valid))
        lr = LogisticRegression()

        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)

        micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        print ('Macro_F1_score:{}'.format(macro_f1))
        print ('Micro_F1_score:{}'.format(micro_f1))


if __name__ == '__main__':

    # print('===== classification =====')
    # train_ratio = [0.2,0.4,0.6,0.8]
    # embeddings_data = '../data/yelpWWW/result/unify_brurb_embedding.mat'
    # print(embeddings_data)
    # exp = Evaluation(embeddings_data)
    # exp.load_embeddings()
    # for t_r in train_ratio:
    #     print(t_r)
    #     exp.classification(train_size=t_r)

    # print('===== link prediction =====')
    # embeddings_data = '../data/yelpWWW_lp/result/unify_0.4bsb+0.6brurb_embedding.mat'
    # print(embeddings_data)
    # exp = Evaluation(embeddings_data)
    # exp.load_embeddings()
    # exp.link_prediction_with_auc()

    # for i in xrange(1,10):
    #     embeddings_data = '../data/yelpWWW_lp/result/unify_'+str(i/10)+'bsb+'+str(1-i/10)+'brurb_embedding.mat'
    #     print(embeddings_data)
    #     exp = Evaluation(embeddings_data)
    #     exp.load_embeddings()
    #     exp.link_prediction_with_auc()

    # print('===== dynamic classification =====')
    # train_ratio = [0.2,0.4,0.6,0.8]
    # embeddings_data = '../data/yelpWWWDynamic/result/0_0.4bsb+0.6brurb_embedding.mat'
    # print(embeddings_data)
    # exp = Evaluation(embeddings_data)
    # exp.load_embeddings()
    # for t_r in train_ratio:
    #     print(t_r)
    #     exp.classification(train_size=t_r)
    #
    # for t_r in train_ratio:
    #     print(t_r)
    #     ma_f1 = []
    #     mi_f1 = []
    #     for i in xrange(0,10):
    #         embeddings_data = '../data/yelpWWWDynamic/result/'+str(i)+'_0.4bsb+0.6brurb_embedding.mat'
    #         print(embeddings_data)
    #         exp = Evaluation(embeddings_data)
    #         exp.load_embeddings()
    #         ma_f1_tmp, mi_f1_tmp = exp.classification(train_size=t_r)
    #         ma_f1.append(ma_f1_tmp)
    #         mi_f1.append(mi_f1_tmp)
    #     print('ave. ma_f1: {}'.format(sum(ma_f1) / 10))
    #     print('ave. mi_f1: {}'.format(sum(mi_f1) / 10))

    # print('===== dynamic link prediction =====')
    # embeddings_data = '../data/yelpWWWDynamic_lp/result/0_0.4bsb+0.6brurb_embedding.mat'
    # print(embeddings_data)
    # exp = Evaluation(embeddings_data)
    # exp.load_embeddings()
    # exp.link_prediction_with_auc()
    # for i in xrange(10):
    #     embeddings_data = '../data/yelpWWWDynamic_lp/result/'+str(i)+'_0.4bsb+0.6brurb_embedding.mat'
    #     print(embeddings_data)
    #     exp = Evaluation(embeddings_data)
    #     exp.load_embeddings()
    #     exp.link_prediction_with_auc()

    # retrain
    print ('YELP')
    print ('retrain...')
    embeddings_data = '../data/yelpWWW/result/unify_0.4bsb+0.6brurb_embedding.mat'
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()
    exp.new_cl()

    # dynamic
    print ('dynamic...')
    embeddings_data = '../data/yelpWWWDynamic/result/10_0.4bsb+0.6brurb_embedding.mat'
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()
    exp.new_cl()