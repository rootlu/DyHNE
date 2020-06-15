# coding: utf-8
# author: lu yf
# create date: 2018/8/11

import os
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir):
        self.data_fold = data_dir
        self.user_list = []
        self.review_list = []
        self.business_list = []
        self.keywords_list = []
        self.stars_list = []

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        with open(os.path.join(self.data_fold, 'user_review.txt')) as ur_file:
            ur_lines = ur_file.readlines()
        for line in ur_lines:
            token = line.strip().split('\t')
            user_name = 'u' + token[0]
            review_name = 'r' + token[1]
            self.user_list.append(user_name)
            self.review_list.append(review_name)

        with open(os.path.join(self.data_fold, 'business_review.txt')) as br_file:
            br_lines = br_file.readlines()
        for line in br_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            review_name = 'r' + token[1]
            self.business_list.append(business_name)
            self.review_list.append(review_name)

        with open(os.path.join(self.data_fold, 'business_stars.txt')) as bs_file:
            bs_lines = bs_file.readlines()
        for line in bs_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            stars_name = 's' + token[1]
            self.business_list.append(business_name)
            self.stars_list.append(stars_name)

        with open(os.path.join(self.data_fold, 'review_keywords.txt')) as rk_file:
            rk_lines = rk_file.readlines()
        for line in rk_lines:
            token = line.strip().split('\t')
            review_name = 'r' + token[0]
            keywords_name = 'k' + token[1]
            self.review_list.append(review_name)
            self.keywords_list.append(keywords_name)

        self.user_list = list(set(self.user_list))
        self.review_list = list(set(self.review_list))
        self.business_list = list(set(self.business_list))
        self.stars_list = list(set(self.stars_list))
        self.keywords_list = list(set(self.keywords_list))

        print ('#users: {}'.format(len(self.user_list)))
        print ('#reviews: {}'.format(len(self.review_list)))
        print ('#businesses: {}'.format(len(self.business_list)))
        print ('#stars: {}'.format(len(self.stars_list)))
        print ('#keywords: {}'.format(len(self.keywords_list)))

        print ('build adj_csr_matrix...')
        row = []
        col = []
        data = np.ones(len(ur_lines))
        for line in ur_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.ur_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.user_list), len(self.review_list)))

        row = []
        col = []
        data = np.ones(len(br_lines))
        for line in br_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.br_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.review_list)))

        row = []
        col = []
        data = np.ones(len(bs_lines))
        for line in bs_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.bs_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.stars_list)))

        row = []
        col = []
        data = np.ones(len(rk_lines))
        for line in rk_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.rk_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.review_list), len(self.keywords_list)))

    def get_data_4_our_symmetrical(self):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical)...')
        bsb_adj_matrix = self.bs_adj_matrix * self.bs_adj_matrix.transpose()
        self.save_mat(bsb_adj_matrix, 'bsb_csr')

        bru_adj_matrix = self.br_adj_matrix * self.ur_adj_matrix.transpose()
        brurb_adj_mtx = bru_adj_matrix * bru_adj_matrix.transpose()
        self.save_mat(brurb_adj_mtx, 'brurb_csr')

        brk_adj_matrix = self.br_adj_matrix * self.rk_adj_matrix
        brkrb_adj_mtx = brk_adj_matrix * brk_adj_matrix.transpose()
        self.save_mat(brkrb_adj_mtx, 'brkrb_csr')

    def save_mat(self,matrix,relation_name):
        scipy.io.savemat(os.path.join('../data/yelpWWW/',relation_name),
                         {relation_name:matrix})


if __name__ == '__main__':
    dh = DataHelper('../data/yelpWWW/oriData/')
    dh.load_data()
    dh.get_data_4_our_symmetrical()

