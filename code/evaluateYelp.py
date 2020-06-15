# coding: utf-8
# author: lu yf
# create date: 2018/7/31

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
        with open('../data/yelp/oriData/business_category.txt', 'r') as author_name_label_file:
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

    def classification(self,train_size):
        x = []
        y = []
        with open('../data/yelp/oriData/business_category.txt', 'r') as author_name_label_file:
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

    def link_prediction_with_hit(self,hist_k):
        print('link prediction...')
        with open('../data/yelp_lp/test_aa_pos_neg.txt','r') as ld_f:
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
        with open('../data/yelp_lp/'+data_type+'_aa_pos.txt', 'r') as aa_pos_f:
            for line in aa_pos_f:
                tokens = line.strip().split('\t')
                if not self.name_emb_dict.has_key(int(tokens[0])) or not self.name_emb_dict.has_key(int(tokens[1])):
                    continue
                pos_1_emb = self.name_emb_dict[int(tokens[0])]
                pos_2_emb = self.name_emb_dict[int(tokens[1])]
                sim_pos = self.calculate_sim(pos_1_emb, pos_2_emb, sum_flag=False)
                x.append(sim_pos)
                y.append(1)
        print('#pos {}: {}'.format(data_type,len(x)))
        with open('../data/yelp_lp/'+data_type+'_aa_neg.txt', 'r') as aa_neg_f:
            for line in aa_neg_f:
                tokens = line.strip().split('\t')
                if not self.name_emb_dict.has_key(int(tokens[0])) or not self.name_emb_dict.has_key(int(tokens[1])):
                    continue
                neg_1_emb = self.name_emb_dict[int(tokens[0])]
                neg_2_emb = self.name_emb_dict[int(tokens[1])]
                sim_neg = self.calculate_sim(neg_1_emb, neg_2_emb, sum_flag=False)
                x.append(sim_neg)
                y.append(0)
        print('#pos+neg {}: {}'.format(data_type,len(x)))
        return x,y

    def link_prediction_with_auc(self):
        train_x, train_y = self.pre_4_link_prediction('train')
        test_x, test_y = self.pre_4_link_prediction('test')
        print('link prediction with auc...')
        print(len(train_x), len(test_x))
        self.binary_classification_aa(train_x, train_y, test_x, test_y)

    def pre_4_node_recommendation(self,author_embeddings,conf_embeddings,author_start_end_id,conf_start_end_id):
        a_embeddings_mat = scipy.io.loadmat(author_embeddings)
        key = filter(lambda k: k.startswith('_') == False, a_embeddings_mat.keys())[0]
        author_embs = a_embeddings_mat[key]
        for i in range(author_start_end_id[0],author_start_end_id[1]):
            self.name_emb_dict['a'+str(i)] = author_embs[i]
        c_embeddings_mat = scipy.io.loadmat(conf_embeddings)
        key = filter(lambda k: k.startswith('_') is False, c_embeddings_mat.keys())[0]
        conf_embs = c_embeddings_mat[key]
        for i in range(conf_start_end_id[0],conf_start_end_id[1]):
            self.name_emb_dict['c'+str(i-conf_start_end_id[0])] = conf_embs[i]

    def node_recommendation(self,hit_k):
        with open('../data/yelp_nr/test_ac_pos_neg.txt', 'r') as test_ac_p_n_f:
            ac_lines = test_ac_p_n_f.readlines()

        hit_or_not = []
        for aa in list(set(ac_lines)):
            ac_list = aa.strip().split('\t')
            pos_a = ac_list[0]
            pos_c = ac_list[1]
            if self.name_emb_dict.has_key('a'+pos_a) and self.name_emb_dict.has_key('c'+pos_c):
                neg_c_list = ac_list[2].split(' ')
                pos_a_embs = self.name_emb_dict['a' + pos_a]
                pos_c_embs = self.name_emb_dict['c' + pos_c]
                pos_sim = self.calculate_sim(pos_a_embs, pos_c_embs, sum_flag=True)
                neg_sim_list = map(lambda x:
                                   self.calculate_sim(pos_a_embs, self.name_emb_dict['c' + x], sum_flag=True),
                                   neg_c_list)
                neg_c_list.append(pos_c)
                neg_sim_list.append(pos_sim)
                sim_dict = dict(zip(neg_c_list, neg_sim_list))
                sorted_sim_triple = sorted(sim_dict.iteritems(), key=lambda d: d[1])
                if (pos_c, pos_sim) in sorted_sim_triple[:hit_k]:
                    hit_or_not.append(1)
                else:
                    hit_or_not.append(0)
        print('#test: {}'.format(len(hit_or_not)))
        print('hit@{}: {}'.format(hit_k, sum(hit_or_not) / len(hit_or_not)))

    def new_cl(self):
        x_train, x_valid, y_train, y_valid = [],[],[],[]
        new_nodes = []
        with open('../dblp_delta_nodes.txt', 'r') as new_node_file:
            for f in new_node_file:
                new_nodes.append(int(f.strip()))

        with open('../data/dblp/oriData/author_label.txt', 'r') as author_name_label_file:
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

    # classification
    embeddings_data = '../data/yelp/result/unify_0.1bsb+0.1bcb+0.8brurb_embedding_0.5g0.5.mat'
    print(embeddings_data)
    exp = Evaluation(embeddings_data)
    exp.load_embeddings()
    exp.kmeans_nmi(cluster_k=5)
    exp.classification(train_size=0.8)

    # # link prediction
    # embeddings_data = '../data/yelp_lp/result/unify_embedding_lp.mat'
    # print(embeddings_data)
    # exp = Evaluation(embeddings_data)
    # exp.load_embeddings()
    # exp.link_prediction_with_auc()

    # embeddings_data = '../data/yelp_nr/result/apc_embedding_nr.mat'
    # # print(embeddings_data)
    # c_start_id = 0
    # exp = Evaluation(embeddings_data)
    # exp.pre_4_node_recommendation(author_embeddings='../data/yelp_nr/result/a_unify_embedding_nr.mat',
    #                               conf_embeddings='../data/yelp_nr/result/apc_embedding_nr.mat',
    #                               author_start_end_id=[0,14475],conf_start_end_id=[14475,14475+20])
    # exp.node_recommendation(hit_k=5)
    # exp.node_recommendation(hit_k=10)
    # exp.node_recommendation(hit_k=15)
