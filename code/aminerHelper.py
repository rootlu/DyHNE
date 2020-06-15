# coding: utf-8
# author: lu yf
# create date: 2018/7/20

import os

import collections
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir):
        self.data_fold = data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []
        self.year_list = []

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print ('loading data...')
        with open(os.path.join(self.data_fold, 'paper_author.txt')) as pa_file:
            self.pa_lines = pa_file.readlines()
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.author_list.append(token[1])
        with open(os.path.join(self.data_fold, 'paper_conf.txt')) as pc_file:
            self.pc_lines = pc_file.readlines()
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.conf_list.append(token[1])
        with open(os.path.join(self.data_fold, 'paper_term.txt')) as pt_file:
            self.pt_lines = pt_file.readlines()
        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.term_list.append(token[1])
        with open(os.path.join(self.data_fold, 'paper_year.txt')) as py_file:
            self.py_lines = py_file.readlines()
        for line in self.py_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.year_list.append(token[1])
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))
        self.year_list = list(set(self.year_list))
        print ('#paper:{}, #author:{}, #conf:{}, #term:{}, #year:{}'.format(len(self.paper_list), len(self.author_list),
                                                                            len(self.conf_list), len(self.term_list),
                                                                            len(self.year_list)))

        print ('build adj_matrix...')
        self.pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pa_adj_matrix[row][col] = 1

        self.pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pc_adj_matrix[row][col] = 1

        self.pt_adj_matrix = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pt_adj_matrix[row][col] = 1

        self.py_adj_matrix = np.zeros([len(self.paper_list), len(self.year_list)], dtype=float)
        for line in self.py_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.py_adj_matrix[row][col] = 1

    def get_data_4_our_symmetrical(self):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical)...')
        print('apa_csr_mtx...')
        apa_adj_matrix = np.matmul(self.pa_adj_matrix.transpose(),self.pa_adj_matrix)
        apa_csr_mtx = csr_matrix(apa_adj_matrix)
        self.save_mat(apa_csr_mtx, 'apa_csr')

        print('apa_csr_mtx...')
        apapa_csr_matrix = np.matmul(apa_adj_matrix,apa_adj_matrix)
        apapa_csr_mtx = csr_matrix(apapa_csr_matrix)
        self.save_mat(apapa_csr_mtx,'apapa_csr')

        print('apcpa_csr_mtx...')
        apc_adj_matrix = np.matmul(self.pa_adj_matrix.transpose(),self.pc_adj_matrix)
        apcpa_adj_mtx = np.matmul(apc_adj_matrix,apc_adj_matrix.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr')

        print('aptpa_csr_mtx...')
        apt_adj_matrix = np.matmul(self.pa_adj_matrix.transpose(), self.pt_adj_matrix)
        aptpa_adj_mtx = np.matmul(apt_adj_matrix, apt_adj_matrix.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr')

        print('apypa_csr_mtx...')
        apy_adj_mtx = np.matmul(self.pa_adj_matrix.transpose(), self.py_adj_matrix)
        apypa_adj_mtx = np.matmul(apy_adj_mtx, apy_adj_mtx.transpose())
        apypa_csr_mtx = csr_matrix(apypa_adj_mtx)
        self.save_mat(apypa_csr_mtx, 'apypa_csr')

        print('pap_csr_mtx...')
        pap_adj_mtx = np.matmul(self.pa_adj_matrix, self.pa_adj_matrix.transpose())
        pap_csr_mtx = csr_matrix(pap_adj_mtx)
        self.save_mat(pap_csr_mtx, 'pap_csr')

        print('pcp_csr_mtx...')
        pcp_adj_mtx = np.matmul(self.pc_adj_matrix,self.pc_adj_matrix.transpose())
        pcp_csr_mtx = csr_matrix(pcp_adj_mtx)
        self.save_mat(pcp_csr_mtx,'pcp_csr')

        print('ptp_csr_mtx...')
        ptp_adj_mtx = np.matmul(self.pt_adj_matrix, self.pt_adj_matrix.transpose())
        ptp_csr_mtx = csr_matrix(ptp_adj_mtx)
        self.save_mat(ptp_csr_mtx, 'ptp_csr')

        print('cpapc_csr_mtx...')
        cpapc_adj_mtx = np.matmul(apc_adj_matrix.transpose(), apc_adj_matrix)
        cpapc_csr_mtx = csr_matrix(cpapc_adj_mtx)
        self.save_mat(cpapc_csr_mtx, 'cpapc_csr')

        print('cptpc_csr_mtx...')
        cpt_adj_mtx = np.matmul(self.pc_adj_matrix.transpose(), self.pt_adj_matrix)
        cptpc_adj_mtx = np.matmul(cpt_adj_mtx, cpt_adj_mtx.transpose())
        cptpc_csr_mtx = csr_matrix(cptpc_adj_mtx)
        self.save_mat(cptpc_csr_mtx, 'cptpc_csr')

        print('cpypc_csr_mtx...')
        cpy_adj_mtx = np.matmul(self.pc_adj_matrix.transpose(),self.py_adj_matrix)
        cpypc_adj_mtx = np.matmul(cpy_adj_mtx, cpy_adj_mtx.transpose())
        cpypc_csr_mtx = csr_matrix(cpypc_adj_mtx)
        self.save_mat(cpypc_csr_mtx, 'cpypc_csr')

    def get_data_4_our_asymmetric(self):
        """
        get data for our model
        deal with asy. metapaths.
        :return:
        """
        print('get data for our (asymmetric)...')
        p = map(lambda x: int(x.strip().split('\t')[0]) + len(self.author_list), self.pa_lines)
        a = map(lambda x: int(x.strip().split('\t')[1]), self.pa_lines)
        row = np.array(a + p)
        col = np.array(p + a)
        data = np.ones(2 * len(self.pa_lines))
        ap_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.author_list) + len(self.paper_list),
                                       len(self.author_list) + len(self.paper_list)))
        self.save_mat(ap_csr_mtx, 'ap_csr')
        self.save_mat(ap_csr_mtx.transpose(), 'pa_csr')

        p = map(lambda x: int(x.strip().split('\t')[0]), self.pc_lines)
        c = map(lambda x: int(x.strip().split('\t')[1]) + len(self.paper_list), self.pc_lines)
        row = np.array(p + c)
        col = np.array(c + p)
        data = np.ones(2 * len(self.pc_lines))
        pc_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.conf_list),
                                       len(self.paper_list) + len(self.conf_list)))
        self.save_mat(pc_csr_mtx, 'pc_csr')

        p = map(lambda x: int(x.strip().split('\t')[0]), self.py_lines)
        y = map(lambda x: int(x.strip().split('\t')[1]) + len(self.paper_list), self.py_lines)
        row = np.array(p + y)
        col = np.array(y + p)
        data = np.ones(2 * len(self.py_lines))
        pt_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.year_list),
                                       len(self.paper_list) + len(self.year_list)))
        self.save_mat(pt_csr_mtx, 'py_csr')

        pc_dict = {}
        for pc in self.pc_lines:
            tokens = pc.strip().split('\t')
            pc_dict[tokens[0]] = tokens[1]  # 1 vs 1
        ac_dict = {}
        for pa in self.pa_lines:
            tokens = pa.strip().split('\t')
            if not ac_dict.has_key(tokens[1]):
                ac_dict[tokens[1]] = []
            if pc_dict.has_key(tokens[0]):
                ac_dict[tokens[1]].append(pc_dict[tokens[0]])
        row = []
        col = []
        data = []
        for a, c_list in ac_dict.items():
            ac_weight = collections.Counter(c_list)
            for c in list(set(c_list)):
                row.append(int(a))
                col.append(int(c) + len(self.author_list))
                data.append(float(ac_weight[c]))
        apc_csr_mtx = csr_matrix((data, (row, col)),
                                 shape=(len(self.conf_list) + len(self.author_list),
                                        len(self.conf_list) + len(self.author_list)))
        self.save_mat(apc_csr_mtx, 'apc_csr')

        py_dict = {}
        for pt in self.py_lines:
            tokens = pt.strip().split('\t')
            if not py_dict.has_key(tokens[0]):
                py_dict[tokens[0]] = []
            py_dict[tokens[0]].append(tokens[1])
        ay_dict = {}
        for pa in self.pa_lines:
            tokens = pa.strip().split('\t')
            if not ay_dict.has_key(tokens[1]):
                ay_dict[tokens[1]] = []
            if py_dict.has_key(tokens[0]):
                for y in py_dict[tokens[0]]:
                    ay_dict[tokens[1]].append(y)
        row = []
        col = []
        data = []
        for a, y_list in ay_dict.items():
            ay_weight = collections.Counter(y_list)
            for y in list(set(y_list)):
                row.append(int(a))
                col.append(int(y) + len(self.author_list))
                data.append(float(ay_weight[y]))
        apy_csr_mtx = csr_matrix((data, (row, col)),
                                 shape=(len(self.year_list) + len(self.author_list),
                                        len(self.year_list) + len(self.author_list)))
        self.save_mat(apy_csr_mtx, 'apy_csr')

    def save_mat(self,matrix,relation_name):
        scipy.io.savemat(os.path.join('../data/aminer/',relation_name),
                         {relation_name:matrix})


if __name__ == '__main__':
    dh = DataHelper('../data/aminer/oriData/')
    dh.load_data()
    dh.get_data_4_our_symmetrical()
    # dh.get_data_4_our_asymmetric()