# coding: utf-8
# user: lu yf
# create date: 2018/8/2

import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix
from tqdm import tqdm


class DataHelper:
    def __init__(self,data_dir):
        self.data_fold = data_dir
        self.user_list = []
        self.review_list = []
        self.business_list = []
        self.city_list = []
        self.state_list = []

        self.user_review = {}
        self.review_user = {}
        self.business_review = {}
        self.review_business = {}
        self.business_city = {}
        self.city_business = {}
        self.business_state = {}
        self.state_business = {}

        self.node2id = {}

        self.train_ur_lines = []
        self.test_ur_lines = []
        self.ur_lines = []
        self.br_line = []
        self.bc_lines = []
        self.bs_lines = []

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        with open(os.path.join(self.data_fold, 'user_review.txt')) as ur_file:
            self.ur_lines = ur_file.readlines()
        for line in self.ur_lines:
            token = line.strip().split('\t')
            user_name = 'u' + token[0]
            review_name = 'r' + token[1]
            self.user_list.append(user_name)
            self.review_list.append(review_name)

        with open(os.path.join(self.data_fold, 'business_review.txt')) as br_file:
            self.br_lines = br_file.readlines()
        for line in self.br_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            review_name = 'r' + token[1]
            self.business_list.append(business_name)
            self.review_list.append(review_name)

        with open(os.path.join(self.data_fold, 'business_city.txt')) as bc_file:
            self.bc_lines = bc_file.readlines()
        for line in self.bc_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            city_name = 'c' + token[1]
            self.business_list.append(business_name)
            self.city_list.append(city_name)

        with open(os.path.join(self.data_fold, 'business_state.txt')) as bs_file:
            self.bs_lines = bs_file.readlines()
        for line in self.bs_lines:
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
        data = np.ones(len(self.ur_lines))
        for line in self.ur_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
        self.ur_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.user_list), len(self.review_list)))

    def split_data(self,load_from_file):
        """
        hidden edges
        load history data from file or not
        :param load_from_file:
        :return:
        """
        if load_from_file:
            print('split data with history data...')
            with open('../data/yelp_lp/train_ur.txt','r') as tr_ur_f:
                self.train_ur_lines = tr_ur_f.readlines()
            with open('../data/yelp_lp/test_ur.txt','r') as te_ur_f:
                self.test_ur_lines = te_ur_f.readlines()
        else:
            print('split data again...')
            self.train_ur_lines = random.sample(self.ur_lines, int(0.8 * len(self.ur_lines)))
            self.test_ur_lines = list(set(self.ur_lines) - set(self.train_ur_lines))
        print('#train ur: {}, #test ur: {}'.format(len(self.train_ur_lines), len(self.test_ur_lines)))

    def process_data(self):
        """
        process data for baselines and our
        get business_user dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        print ('process data...')

        train_row = []
        train_col = []
        train_data = np.ones(len(self.train_ur_lines))
        test_row = []
        test_col = []
        test_data = np.ones(len(self.test_ur_lines))
        for line in self.ur_lines:
            token = line.strip('\n').split('\t')
            user_name = 'u' + token[0]
            review_name = 'r' + token[1]
            if not self.node2id.has_key(user_name):
                self.node2id[user_name] = len(self.node2id)
            if not self.node2id.has_key(review_name):
                self.node2id[review_name] = len(self.node2id)

            if line in self.train_ur_lines:
                if not self.user_review.has_key(user_name):
                    self.user_review[user_name] = set()
                self.user_review[user_name].add(review_name)
                if not self.review_user.has_key(review_name):
                    self.review_user[review_name] = set()
                self.review_user[review_name].add(user_name)
                train_row.append(int(token[0]))
                train_col.append(int(token[1]))
            elif line in self.test_ur_lines:
                test_row.append(int(token[0]))
                test_col.append(int(token[1]))

        self.train_ur_adj_matrix = csr_matrix((train_data, (train_row, train_row)),
                                              shape=(len(self.user_list), len(self.review_list)))
        self.test_ur_adj_matrix = csr_matrix((test_data, (test_row, test_col)),
                                             shape=(len(self.user_list), len(self.review_list)))

        row = []
        col = []
        data = np.ones(len(self.br_line))
        for line in self.br_line:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))

            business_name = 'b' + token[0]
            review_name = 'r' + token[1]
            if not self.business_review.has_key(business_name):
                self.business_review[business_name] = set()
            self.business_review[business_name].add(review_name)
            if not self.review_business.has_key(review_name):
                self.review_business[review_name] = set()
            self.review_business[review_name].add(business_name)
            if not self.node2id.has_key(business_name):
                self.node2id[business_name] = len(self.node2id)
            if not self.node2id.has_key(review_name):
                self.node2id[review_name] = len(self.node2id)
        self.br_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.review_list)))

        row = []
        col = []
        data = np.ones(len(self.bc_lines))
        for line in self.bc_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
            business_name = 'b' + token[0]
            city_name = 'c' + token[1]
            if not self.business_city.has_key(business_name):
                self.business_city[business_name] = set()
            self.business_city[business_name].add(city_name)
            if not self.city_business.has_key(city_name):
                self.city_business[city_name] = set()
            self.city_business[city_name].add(business_name)
            if not self.node2id.has_key(business_name):
                self.node2id[business_name] = len(self.node2id)
            if not self.node2id.has_key(city_name):
                self.node2id[city_name] = len(self.node2id)
        self.bc_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.city_list)))

        row = []
        col = []
        data = np.ones(len(self.bs_lines))
        for line in self.bs_lines:
            token = line.strip('\n').split('\t')
            row.append(int(token[0]))
            col.append(int(token[1]))
            business_name = 'b' + token[0]
            state_name = 's' + token[1]
            if not self.business_state.has_key(business_name):
                self.business_state[business_name] = set()
            self.business_state[business_name].add(state_name)
            if not self.state_business.has_key(state_name):
                self.state_business[state_name] = set()
            self.state_business[state_name].add(business_name)
            if not self.node2id.has_key(business_name):
                self.node2id[business_name] = len(self.node2id)
            if not self.node2id.has_key(state_name):
                self.node2id[state_name] = len(self.node2id)
        self.bs_adj_matrix = csr_matrix((data, (row, col)),
                                        shape=(len(self.business_list), len(self.state_list)))

        train_bru_adj_mtx = self.br_adj_matrix * self.train_ur_adj_matrix.transpose()
        self.train_brurb_adj_mtx = train_bru_adj_mtx * train_bru_adj_mtx.transpose()
        test_bru_adj_mtx = self.br_adj_matrix * self.test_ur_adj_matrix.transpose()
        self.test_brurb_adj_mtx = test_bru_adj_mtx * test_bru_adj_mtx.transpose()
        bru_adj_mtx = self.br_adj_matrix * self.ur_adj_matrix.transpose()
        self.brurb_adj_mtx = bru_adj_mtx * bru_adj_mtx.transpose()

    def save_train_test_ur_data(self):
        """
        save business-user in train set and test set
        save user co-appear in train and test
        :return:
        """
        with open('../data/yelp_lp/train_ur.txt','w') as train_ur_f:
            for line in self.train_ur_lines:
                train_ur_f.write(line)
        with open('../data/yelp_lp/test_ur.txt','w') as test_ur_f:
            for line in self.test_ur_lines:
                test_ur_f.write(line)

    def get_brurb_pos_neg_data(self,data_type):
        """
        get positive and negative co-user data for link prediction
        :param data_type: train or test
        :return:
        """
        print('get {} brubr data...'.format(data_type))
        if data_type == 'train':
            brurb_adj_mtx = self.train_brurb_adj_mtx
        elif data_type == 'test':
            brurb_adj_mtx = self.test_brurb_adj_mtx
        row, col = np.nonzero(brurb_adj_mtx)
        print('#{} brurb: {}'.format(data_type,len(row)))
        with open('../data/yelp_lp/'+data_type+'_bb_pos.txt', 'w') as bb_pos_f:
            for i in xrange(len(row)):
                pos_b_1 = row[i]
                pos_b_2 = col[i]
                bb_pos_f.write(str(pos_b_1)+'\t'+str(pos_b_2)+'\n')

        # negative brurb
        zero_bb = np.where(self.brurb_adj_mtx.todense() == 0)
        cand_bb_list = zip(zero_bb[0],zero_bb[1])
        neg_bb_list = random.sample(cand_bb_list,len(row))

        with open('../data/yelp_lp/'+data_type+'_bb_neg.txt', 'w') as bb_neg_f:
            for neg_bb in neg_bb_list:
                bb_neg_f.write(str(neg_bb[0]) + '\t' + str(neg_bb[1]) + '\n')

    def get_data_4_baselines(self):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec ...')
        with open(os.path.join('../baseline/yelp_lp/', 'dw.edgelist_lp'), 'w') as edge_file:
            for u, r_list in self.user_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[u]) + ' ' + str(self.node2id[r]) + '\n')
                    edge_file.write(str(self.node2id[r]) + ' ' + str(self.node2id[u]) + '\n')
            for b, r_list in self.business_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[r]) + '\n')
                    edge_file.write(str(self.node2id[r]) + ' ' + str(self.node2id[b]) + '\n')
            for c, b_list in self.city_business.items():
                for b in b_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[c]) + '\n')
                    edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[b]) + '\n')
            for s, b_list in self.state_business.items():
                for b in b_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[s]) + '\n')
                    edge_file.write(str(self.node2id[s]) + ' ' + str(self.node2id[b]) + '\n')

        with open(os.path.join('../baseline/yelp_lp/', 'dw.node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for line
        print ('get data for line ...')
        with open(os.path.join('../baseline/yelp_lp/', 'line.edgelist_lp'), 'w') as edge_file:
            for u, r_list in self.user_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[u]) + ' ' + str(self.node2id[r]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[r]) + ' ' + str(self.node2id[u]) + ' ' + str(1) + '\n')
            for b, r_list in self.business_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[r]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[r]) + ' ' + str(self.node2id[b]) + ' ' + str(1) + '\n')
            for c, b_list in self.city_business.items():
                for b in b_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[c]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[b]) + ' ' + str(1) + '\n')
            for s, b_list in self.state_business.items():
                for b in b_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[s]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[s]) + ' ' + str(self.node2id[b]) + ' ' + str(1) + '\n')

        with open(os.path.join('../baseline/yelp_lp/', 'line.node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for esim
        print('get data for esim...')
        with open(os.path.join('../baseline/yelp_lp/', 'esim.metapath_lp'), 'w') as metapath_file:
            metapath_file.write('bsb 0.1' + '\n')
            metapath_file.write('bcb 0.8' + '\n')
            metapath_file.write('brurb 0.1' + '\n')
            # metapath_file.write('apypa 0.2')
            metapath_file.write('\n')
        with open(os.path.join('../baseline/yelp_lp/', 'esim.node_lp'), 'w') as node_file:
            for u in list(self.user_review.keys()):
                node_file.write(u + ' u')
                node_file.write('\n')
            for r in self.review_list:
                node_file.write('r' + r + ' r')
                node_file.write('\n')
            for b in self.business_list:
                node_file.write('b' + b + ' b')
                node_file.write('\n')
            for c in self.city_list:
                node_file.write('c' + c + ' c')
                node_file.write('\n')
            for s in self.state_list:
                node_file.write('s' + s + ' s')
                node_file.write('\n')
        with open(os.path.join('../baseline/yelp_lp/', 'esim.link_lp'), 'w') as net_file:
            for u, r_list in self.user_review.items():
                for r in r_list:
                    net_file.write(u + ' ')
                    net_file.write(r + ' ')
                    net_file.write('\n')
                    net_file.write(r + ' ')
                    net_file.write(u + ' ')
                    net_file.write('\n')
            for b, r_list in self.business_review.items():
                for r in r_list:
                    net_file.write(b + ' ')
                    net_file.write(r + ' ')
                    net_file.write('\n')
                    net_file.write(r + ' ')
                    net_file.write(b + ' ')
                    net_file.write('\n')
            for c, b_list in self.city_business.items():
                for b in b_list:
                    net_file.write(b + ' ')
                    net_file.write(c + ' ')
                    net_file.write('\n')
                    net_file.write(c + ' ')
                    net_file.write(b + ' ')
                    net_file.write('\n')
            for s, b_list in self.state_business.items():
                for b in b_list:
                    net_file.write(b + ' ')
                    net_file.write(s + ' ')
                    net_file.write('\n')
                    net_file.write(s + ' ')
                    net_file.write(b + ' ')
                    net_file.write('\n')

    def get_data_4_m2v(self, metapath, num_walks, walk_length):
        """
        get data for metapath2vec
        over all authors!!
        :param metapath:
        :param num_walks:
        :param walk_length:
        :return:
        """
        # data for metapath2vec
        print('get data for metapath2vec')
        print ('generating paths randomly via {}...'.format(metapath))
        file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + str(walk_length) + '_paths_lp.txt'
        outfile = open(os.path.join('../baseline/yelp_lp/', file_name), 'w')
        if metapath == 'bsb':
            for business in tqdm(self.business_state):
                for j in xrange(0,num_walks):
                    outline = business
                    for i in xrange(0,walk_length):
                        if self.business_state.has_key(business):
                            next_s_list = list(self.business_state[business])
                            next_s_node = random.choice(next_s_list)
                        else: continue
                        utline += ' ' + next_s_node

                        if self.state_business.has_key(next_s_node):
                            next_b_list = list(self.state_business[next_s_node])
                            next_b_node = random.choice(next_b_list)
                        else: continue
                        outline += ' ' + next_b_node
                        business = next_b_node
        elif metapath == 'bcb':
            for business in tqdm(self.business_city):
                for j in xrange(0, num_walks):
                    outline = business
                    for i in xrange(0, walk_length):
                        if self.business_city.has_key(business):
                            next_c_list = list(self.business_city[business])
                            next_c_node = random.choice(next_c_list)
                        else: continue
                        outline += ' ' + next_c_node

                        if self.city_business.has_key(next_c_node):
                            next_b_list = list(self.city_business[next_c_node])
                            next_b_node = random.choice(next_b_list)
                        else: continue
                        outline += ' ' + next_b_node
                        business = next_b_node
        elif metapath == 'brurb':
            for business in tqdm(self.business_review):
                for j in xrange(0, num_walks):
                    outline = business
                    for i in xrange(0, walk_length):
                        if self.business_review.has_key(business):
                            next_r_list = list(self.business_review[business])
                            next_r_node = random.choice(next_r_list)
                        else: continue
                        outline += ' ' + next_r_node

                        if self.review_user.has_key(next_r_node):
                            next_u_list = list(self.review_user[next_r_node])
                            next_u_node = random.choice(next_u_list)
                        else: continue
                        outline += ' ' + next_u_node

                        if self.user_review.has_key(next_u_node):
                            next_r_list = list(self.user_review[next_u_node])
                            next_r_node = random.choice(next_r_list)
                        else: continue
                        outline += ' ' + next_r_node

                        if self.review_business.has_key(next_r_node):
                            next_b_list = list(self.review_business[next_r_node])
                            next_b_node = random.choice(next_b_list)
                        else: continue
                        outline += ' ' + next_b_node
                        business = next_b_node

        outfile.close()

    def get_data_4_our_symmetrical(self):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical)...')

        self.save_mat(self.train_brurb_adj_mtx, 'brurb_csr_lp')

        bsb_csr_mtx = self.bs_adj_matrix * self.bs_adj_matrix.transpose()
        self.save_mat(bsb_csr_mtx,'bsb_csr_lp')

        bcb_csr_mtx = self.bc_adj_matrix * self.bc_adj_matrix.transpose()
        self.save_mat(bcb_csr_mtx,'bcb_csr_mtx')

    def save_mat(self, matrix, relation_name):
        """
        save data to mat
        :param matrix:
        :param relation_name:
        :return:
        """
        scipy.io.savemat(os.path.join('../data/yelp_lp/', relation_name),
                         {relation_name: matrix})


if __name__ == '__main__':
    dh = DataHelper('../data/yelp/oriData/')
    dh.load_data()

    dh.split_data(load_from_file=False)
    dh.process_data()

    dh.save_train_test_ur_data()

    # # link prediction
    dh.get_data_4_our_symmetrical()

    dh.get_data_4_baselines()
    dh.get_data_4_m2v('apa', 10, 50)
    dh.get_data_4_m2v('apcpa', 10, 50)
    dh.get_data_4_m2v('aptpa', 10, 50)
    # dh.get_data_4_m2v('apa',10,100)
    # dh.get_data_4_m2v('apcpa',10,100)
    # dh.get_data_4_m2v('aptpa',10,100)

    dh.get_brurb_pos_neg_data(data_type='train')
    dh.get_brurb_pos_neg_data(data_type='test')

