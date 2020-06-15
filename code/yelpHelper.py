# coding: utf-8
# author: lu yf
# create date: 2018/7/31

import os
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir):
        self.yelp_data_fold = data_dir
        self.data_fold = '../data/yelp/oriData/'
        self.user_list = []
        self.review_list = []
        self.business_list = []
        self.state_list = []
        self.city_list = []

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

        with open(os.path.join(self.data_fold, 'business_city.txt')) as bc_file:
            bc_lines = bc_file.readlines()
        for line in bc_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            city_name = 'c' + token[1]
            self.business_list.append(business_name)
            self.city_list.append(city_name)

        with open(os.path.join(self.data_fold, 'business_state.txt')) as bs_file:
            bs_lines = bs_file.readlines()
        for line in bs_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            state_name = 's' + token[1]
            self.business_list.append(business_name)
            self.state_list.append(state_name)

        self.user_list = list(set(self.user_list))
        self.review_list = list(set(self.review_list))
        self.business_list = list(set(self.business_list))
        self.city_list = list(set(self.city_list))
        self.state_list = list(set(self.state_list))

        print ('#users: {}'.format(len(self.user_list)))
        print ('#reviews: {}'.format(len(self.review_list)))
        print ('#businesses: {}'.format(len(self.business_list)))
        print ('#cities: {}'.format(len(self.city_list)))
        print('#states: {}'.format(len(self.state_list)))

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
        data = np.ones(len(bc_lines))
        for line in bc_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.bc_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.city_list)))

        row = []
        col = []
        data = np.ones(len(bs_lines))
        for line in bs_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.bs_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.state_list)))

    def get_data_4_our_symmetrical(self):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical)...')
        bcb_adj_matrix = self.bc_adj_matrix * self.bc_adj_matrix.transpose()
        self.save_mat(bcb_adj_matrix, 'bcb_csr')

        bsb_csr_matrix = self.bs_adj_matrix * self.bs_adj_matrix.transpose()
        self.save_mat(bsb_csr_matrix,'bsb_csr')

        bru_adj_matrix = self.br_adj_matrix * self.ur_adj_matrix.transpose()
        brurb_adj_mtx = bru_adj_matrix * bru_adj_matrix.transpose()
        self.save_mat(brurb_adj_mtx, 'brurb_csr')

    def save_mat(self,matrix,relation_name):
        scipy.io.savemat(os.path.join(self.yelp_data_fold,relation_name),
                         {relation_name:matrix})


if __name__ == '__main__':
    dh = DataHelper('../data/yelp/oriData/')
    dh.load_data()
    dh.get_data_4_our_symmetrical()

