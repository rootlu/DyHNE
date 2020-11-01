# coding:utf-8
# author: lu yf
# create date: 2018/6/25

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
        key = filter(lambda k:k.startswith('_') == False,embeddings_mat.keys())[0]

        embeddings = embeddings_mat[key]
        for i in range(len(embeddings)):
            self.name_emb_dict[i] = embeddings[i]

    def kmeans_nmi(self,cluster_k):
        x = []
        y = []
        with open('../data/dblp/oriData/author_label.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))

        km = KMeans(n_clusters=cluster_k)
        km.fit(x, y)
        y_pre = km.predict(x)
        # y_pre = km.fit_predict(x,y)
        nmi = normalized_mutual_info_score(y, y_pre)
        print('Kmean, k={}, nmi={}'.format(cluster_k, nmi))

    def classification(self,train_size):
        x = []
        y = []
        with open('../data/dblp/oriData/author_label.txt', 'r') as author_name_label_file:
            author_name_label_lines = author_name_label_file.readlines()
        for line in author_name_label_lines:
            tokens = line.strip().split('\t')
            if self.name_emb_dict.has_key(int(tokens[0])):
                x.append(list(self.name_emb_dict[int(tokens[0])]))
                y.append(int(tokens[1]))

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1-train_size,random_state=9)
        # print ('train_size: {}'.format(train_sicv ze))
        lr = LogisticRegression()

        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)

        micro_f1 = f1_score(y_valid, y_valid_pred,average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred,average='macro')
        print ('Macro_F1_score:{}'.format(macro_f1))
        print ('Micro_F1_score:{}'.format(micro_f1))

    def clustering_visual(self,x,y,title):
        if not os.path.exists('./figures'):
            os.mkdir('./figures')
        tsne = TSNE(n_components=2)
        Y = tsne.fit_transform(x)
        # with open(os.path.join('./figures',title+'.pickle'), 'wb') as f:
        #     pickle.dump([x,Y,y],f)
        # plt.title(title)
        plt.scatter(Y[:, 0],Y[:, 1],c=y, marker='.')
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.legend()
        plt.axis('off')  # 关闭xy坐标轴
        plt.savefig(os.path.join('./figures',title + '.eps'),fomat='eps')
        plt.savefig(os.path.join('./figures',title + '.png'))
        plt.clf()

    def link_prediction(self,hist_k):
        print('link prediction...')
        with open('../data/dblp/oriData/test_aa_pos_neg.txt','r') as ld_f:
            apa_lines = ld_f.readlines()
        hist_or_not = 0
        for aa in tqdm(list(set(apa_lines))):
            a_list = aa.strip().split('\t')
            pos_a_1 = a_list[0]
            pos_a_2 = a_list[1]
            neg_a_list = a_list[2].split(' ')
            pos_a_1_embs = self.name_emb_dict[int(pos_a_1)]
            pos_a_2_embs = self.name_emb_dict[int(pos_a_2)]
            pos_sim = self.calculate_sim(pos_a_1_embs,pos_a_2_embs,'lp')
            neg_sim_list = map(lambda x:
                               self.calculate_sim(pos_a_1_embs,self.name_emb_dict[int(x)],'lp'),
                               neg_a_list)
            neg_a_list.append(pos_a_2)
            neg_sim_list.append(pos_sim)
            sim_dict = dict(zip(neg_a_list,neg_sim_list))
            sorted_sim_triple = sorted(sim_dict.iteritems(),key=lambda d:d[1])
            if (pos_a_2,pos_sim) in sorted_sim_triple[:hist_k]:
                hist_or_not += 1
            else:
                hist_or_not += 0
        print('hit@{}: {}'.format(hist_k,hist_or_not/len(set(apa_lines))))

    def calculate_sim(self,u,v,flag):
        if flag == 'lp':
            return sum(np.abs(np.array(u)-np.array(v)))
        elif flag == 'nr':
            return np.abs(np.array(u)-np.array(v))
        # return sum(np.array(u)*np.array(v))

    def node_recommendation(self,train_data,test_data,relation_type):
        print('node recommendation for {}...'.format(relation_type))
        with open(train_data, 'r') as ld_f:
            train_ac_lines = ld_f.readlines()
        train_x = []
        train_y = []
        for tr_ac in train_ac_lines:
            tokens = tr_ac.strip().split('\t')
            pos_a = int(tokens[0])
            if relation_type == 'aa':
                pos_c = int(tokens[1])
                neg_c = int(tokens[2].split(' ')[0])
            elif relation_type == 'ac':
                pos_c = int(tokens[1])+14475
                neg_c = int(tokens[2].split(' ')[0])+14475

            sim_pos_a_pos_c = self.calculate_sim(
                self.name_emb_dict[pos_a],self.name_emb_dict[pos_c],'nr')
            sim_pos_a_neg_c = self.calculate_sim(
                self.name_emb_dict[pos_a],self.name_emb_dict[neg_c],'nr')
            train_x.append(sim_pos_a_pos_c)
            train_y.append(1)
            train_x.append(sim_pos_a_neg_c)
            train_y.append(0)

        with open(test_data, 'r') as ld_f:
            test_aca_lines = ld_f.readlines()
        test_x = []
        test_y = []
        for te_ac in test_aca_lines:
            tokens = te_ac.strip().split('\t')
            pos_a = int(tokens[0])
            if relation_type == 'aa':
                pos_c = int(tokens[1])
                neg_c = int(tokens[2].split(' ')[0])
            elif relation_type == 'ac':
                pos_c = int(tokens[1]) + 14475
                neg_c = int(tokens[2].split(' ')[0]) + 14475
            sim_pos_a_pos_c = self.calculate_sim(
                self.name_emb_dict[pos_a], self.name_emb_dict[pos_c], 'nr')
            sim_pos_a_neg_c = self.calculate_sim(
                self.name_emb_dict[pos_a], self.name_emb_dict[neg_c], 'nr')
            test_x.append(sim_pos_a_pos_c)
            test_y.append(1)
            test_x.append(sim_pos_a_neg_c)
            test_y.append(0)

        self.binary_classification_aa(train_x,train_y,test_x,test_y)

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


if __name__ == '__main__':

    embeddings_data = '../data/dblp/result/split_embedding_lp.mat'
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()

    exp.kmeans_nmi(cluster_k=4)
    exp.classification(train_size=0.8)

    # node recommendation for a-a
    train_aa_data = '../data/dblp/oriData/train_aa_pos_neg.txt'
    test_aa_data = '../data/dblp/oriData/test_aa_pos_neg.txt'
    # node recommendation for a-c
    train_ac_data = '../data/dblp/oriData/train_ac_pos_neg.txt'
    test_ac_data = '../data/dblp/oriData/test_ac_pos_neg.txt'

    exp.link_prediction(hist_k=5)
    exp.node_recommendation(train_aa_data,test_aa_data,'aa')
    exp.node_recommendation(train_ac_data,test_ac_data,'ac')
