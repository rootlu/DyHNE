# coding: utf-8
# author: lu yf
# create date: 2018/8/9

import os
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir,our_data_dir):
        self.dblp_data_fold = data_dir
        self.our_data_fold = our_data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []
        self.paper_num = 14376
        self.author_num = 14475
        self.term_num = 8811
        self.conf_num = 20

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print ('loading data...')
        with open(os.path.join('../data/dblp_lp_cikm', 'paper_author_train.txt')) as pa_file:
            pa_lines = pa_file.readlines()
        for line in pa_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.author_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_conf.txt')) as pc_file:
            pc_lines = pc_file.readlines()
        for line in pc_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.conf_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_term.txt')) as pt_file:
            pt_lines = pt_file.readlines()
        for line in pt_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.term_list.append(token[1])
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))
        print ('#paper:{}, #author:{}, #conf:{}, term:{}'.format(len(self.paper_list), len(self.author_list),
                                                                 len(self.conf_list), len(self.term_list)))

        print ('build adj_matrix...')
        pa_adj_matrix = np.zeros([self.paper_num, self.author_num], dtype=float)
        for line in pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1

        pc_adj_matrix = np.zeros([self.paper_num, self.conf_num], dtype=float)
        for line in pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1

        pt_adj_matrix = np.zeros([self.paper_num, self.term_num], dtype=float)
        for line in pt_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pt_adj_matrix[row][col] = 1

        ap_adj_matrix = np.transpose(pa_adj_matrix)
        ac_adj_matrix = np.matmul(np.transpose(pa_adj_matrix), pc_adj_matrix)
        at_adj_matrix = np.matmul(ap_adj_matrix, pt_adj_matrix)
        apa_adj_matrix = np.matmul(ap_adj_matrix,ap_adj_matrix.transpose())
        aca_adj_matrix = np.matmul(ac_adj_matrix,ac_adj_matrix.transpose())
        ata_adj_matrix = np.matmul(at_adj_matrix,at_adj_matrix.transpose())

        print('save matrix...')
        self.save_mat(apa_adj_matrix,'apa_csr_lp')
        self.save_mat(aca_adj_matrix,'apcpa_csr_lp')
        self.save_mat(ata_adj_matrix,'aptpa_csr_lp')

    def save_mat(self,matrix,relation_name):
        csr_mtx = csr_matrix(matrix)
        scipy.io.savemat(os.path.join(self.our_data_fold,relation_name),
                         {relation_name:csr_mtx})


if __name__ == '__main__':
    dh = DataHelper(data_dir='../data/dblp/oriData/',
                    our_data_dir='../data/dblp_lp_cikm/')
    dh.load_data()

