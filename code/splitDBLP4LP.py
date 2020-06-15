# coding: utf-8
# author: lu yf
# create date: 2018/8/9

from __future__ import division
import os
import numpy as np
import random
from scipy.sparse import csr_matrix
from tqdm import tqdm
import csv


class DBLP4LinkPrediction:
    def __init__(self,data_dir,our_dir):
        self.dblp_data_fold = data_dir
        self.our_data_fold = our_dir

        self.author_list = []
        self.paper_list = []
        self.paper_author = {}
        self.author_paper = {}

    def process_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print('process data...')
        with open(os.path.join(self.dblp_data_fold, 'paper_author.txt')) as pa_file:
            pa_lines = pa_file.readlines()
        for line in pa_lines:
            token = line.strip().split('\t')
            paper_name = token[0]
            author_name = token[1]
            self.author_list.append(author_name)
            self.paper_list.append(paper_name)
            if not self.paper_author.has_key(paper_name):
                self.paper_author[paper_name] = set()
            self.paper_author[paper_name].add(author_name)
            if not self.author_paper.has_key(author_name):
                self.author_paper[author_name] = set()
            self.author_paper[author_name].add(paper_name)

        self.author_list = list(set(self.author_list))
        self.paper_list = list(set(self.paper_list))

        test_paper_list = random.sample(self.paper_list,int(len(self.paper_list)*0.2))
        train_paper_list = list(set(self.paper_list) - set(test_paper_list))

        train_pa_set = set()
        test_pa_set = set()
        for p, a_list in self.paper_author.items():
            a_list = list(set(a_list))
            if p in train_paper_list:
                for a in a_list:
                    train_pa_set.add((p,a))

            elif p in test_paper_list:
                for a in a_list:
                    test_pa_set.add((p,a))

        print('#train pa: {}'.format(len(train_pa_set)))
        print('#test pa: {}'.format(len(test_pa_set)))

        with open(os.path.join(self.our_data_fold,'paper_author_train.txt'),'w') as pa_lp_train_file:
            for pa in train_pa_set:
                pa_lp_train_file.write(str(pa[0]) + '\t' + str(pa[1]) + '\n')

        with open(os.path.join(self.our_data_fold,'paper_author_test.txt'),'w') as pa_lp_test_file:
            for pa in test_pa_set:
                pa_lp_test_file.write(str(pa[0]) + '\t' + str(pa[1]) + '\n')

        self.get_aa_data('train',train_pa_set)
        self.get_aa_data('test',test_pa_set)

    def get_aa_data(self,data_type,pa_data):
        print('get aa...')
        paper_author = {}
        author_set = set()
        for pa in pa_data:
            paper_name = pa[0]
            author_name = pa[1]
            author_set.add(author_name)
            if not paper_author.has_key(paper_name):
                paper_author[paper_name] = set()
            paper_author[paper_name].add(author_name)

        print('#author in {}: {}'.format(data_type,len(author_set)))
        coauthor_set = set()
        for p, a_list in paper_author.items():
            a_list = list(set(a_list))
            for i in xrange(0,len(a_list)-1):
                for j in xrange(1,len(a_list)):
                    if (a_list[i], a_list[j]) not in coauthor_set and \
                            (a_list[j],a_list[i]) not in coauthor_set:
                        coauthor_set.add((a_list[i],a_list[j]))

        with open(os.path.join(self.our_data_fold,'pos_coauthor_'+data_type),'w') as coa_lp_pos_file:
            for pa in coauthor_set:
                coa_lp_pos_file.write(str(pa[0]) + '\t' + str(pa[1]) + '\n')

        with open(os.path.join(self.our_data_fold,'neg_coauthor_'+data_type),'w') as coa_lp_neg_file:
            neg_coauthor_set = set()
            for i in tqdm(xrange(len(coauthor_set))):
                while 1:
                    neg_a_1 = random.choice(list(author_set))
                    neg_a_2 = random.choice(list(author_set))
                    if ((neg_a_1, neg_a_2) not in coauthor_set) and \
                            ((neg_a_2, neg_a_1) not in coauthor_set) and (
                            neg_a_1 != neg_a_2):
                        neg_coauthor_set.add((neg_a_1,neg_a_2))
                        break
            for n_c in neg_coauthor_set:
                coa_lp_neg_file.write(str(n_c[0]) + '\t' + str(n_c[1]) + '\n')


if __name__ == "__main__":
    data_helper = DBLP4LinkPrediction(data_dir='../data/dblp/oriData/',
                                      our_dir='../data/dblp_lp_cikm/')
    data_helper.process_data()

