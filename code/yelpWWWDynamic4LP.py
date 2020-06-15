# coding: utf-8
# author: lu yf
# create date: 2018/8/13

import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir,baseline_dir,our_dir):
        self.data_fold = data_dir
        self.baseline_fold = baseline_dir
        self.our_fold = our_dir
        self.user_list = []
        self.review_list = []
        self.business_list = []
        self.stars_list = []

        self.node2id = {}

        self.business_stars = {}
        self.stars_business = {}

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

        with open(os.path.join('../data/yelpWWW_lp/', 'business_review_train.txt')) as br_file:
            self.br_lines = br_file.readlines()
        for line in self.br_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            review_name = 'r' + token[1]
            self.business_list.append(business_name)
            self.review_list.append(review_name)

        with open(os.path.join(self.data_fold, 'business_stars.txt')) as bs_file:
            self.bs_lines = bs_file.readlines()
        for line in self.bs_lines:
            token = line.strip().split('\t')
            business_name = 'b' + token[0]
            stars_name = 's' + token[1]
            self.business_list.append(business_name)
            self.stars_list.append(stars_name)

        self.user_list = list(set(self.user_list))
        self.review_list = list(set(self.review_list))
        self.business_list = list(set(self.business_list))
        self.stars_list = list(set(self.stars_list))

        print ('#users: {}'.format(len(self.user_list)))
        print ('#reviews: {}'.format(len(self.review_list)))
        print ('#businesses: {}'.format(len(self.business_list)))
        print ('#stars: {}'.format(len(self.stars_list)))

        self.load_node2id()
        print('#nodes: {}'.format(len(self.node2id)))

        self.bs_adj_mtx = np.zeros([len(self.business_list), len(self.stars_list)], dtype=float)
        for line in self.bs_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.bs_adj_mtx[row][col] = 1
            business_name = 'b' + token[0]
            stars_name = 's' + token[1]
            if not self.business_stars.has_key(business_name):
                self.business_stars[business_name] = set()
            self.business_stars[business_name].add(stars_name)
            if not self.stars_business.has_key(stars_name):
                self.stars_business[stars_name] = set()
            self.stars_business[stars_name].add(business_name)

    def load_node2id(self):
        with open(os.path.join('../baseline/yelpWWW/', 'node2id'), 'r') as n2id_file:
            for line in n2id_file:
                tokens = line.strip().split('\t')
                self.node2id[tokens[0]] = int(tokens[1])

    def split_data(self,load_from_file):
        """
        hidden edges
        load history data from file or not
        :param load_from_file:
        :return:
        """
        if load_from_file:
            pass
        else:
            print('split data again...')
            total_review = len(self.review_list)
            current_review_list = self.review_list
            for i in xrange(10):  # time steps == 10
                delta_review = random.sample(current_review_list,int(0.001*total_review))
                current_review_list = list(set(current_review_list)-set(delta_review))

                current_data_dict = self.process_data(current_review_list,10-i-1)  # start from t0 to t9, t10 is all nodes

                # self.get_data_4_our_symmetrical(current_data_dict, 10 - i - 1)
                # self.get_data_4_dane(current_data_dict,10-i-1)
                # if i == 9:  # static network
                #     self.get_data_4_baselines(current_data_dict,10-i-1)
                #     self.get_data_4_m2v('bsb',10,50,current_data_dict,10-i-1)
                #     self.get_data_4_m2v('brurb',10,50,current_data_dict,10-i-1)

                print('#time step: {}, #delta review: {}'.format(10-i-1, len(delta_review)))
                print('#current review: {}'.format(len(current_review_list)))

    def process_data(self,current_review_list,time_step):
        """
        process data for baselines and our
        get paper_author dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        business_review = {}
        review_business = {}
        user_review = {}
        review_user = {}

        print ('process data at time step: {}'.format(time_step))
        ur_adj_mtx = np.zeros([len(self.user_list), len(self.review_list)], dtype=float)
        with open(os.path.join('../data/yelpWWWDynamic_lp/oriData', 'user_review'+str(time_step)+'.txt'),'w') as pa_file:
            for line in self.ur_lines:
                token = line.strip('\n').split('\t')
                user_name = 'u' + token[0]
                review_name = 'r' + token[1]
                if review_name in current_review_list:
                    pa_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    ur_adj_mtx[row][col] = 1
                    if not user_review.has_key(user_name):
                        user_review[user_name] = set()
                    user_review[user_name].add(review_name)
                    if not review_user.has_key(review_name):
                        review_user[review_name] = set()
                    review_user[review_name].add(user_name)

        br_adj_mtx = np.zeros([len(self.business_list), len(self.review_list)], dtype=float)
        with open(os.path.join('../data/yelpWWWDynamic_lp/oriData', 'business_review'+str(time_step)+'.txt'),'w') as pc_file:
            for line in self.br_lines:
                token = line.strip('\n').split('\t')
                business_name = 'b' + token[0]
                review_name = 'r' + token[1]
                if review_name in current_review_list:
                    pc_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    br_adj_mtx[row][col] = 1
                    if not business_review.has_key(business_name):
                        business_review[business_name] = set()
                    business_review[business_name].add(review_name)
                    if not review_business.has_key(review_name):
                        review_business[review_name] = set()
                    review_business[review_name].add(business_name)

        return {'ur':user_review,'ru':review_user,
                'br':business_review,'rb':review_business,
                'ur_adj_mtx':ur_adj_mtx,'br_adj_mtx':br_adj_mtx}

    def get_data_4_baselines(self,current_data_dict,time_step):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        user_review = current_data_dict['ur']
        business_review = current_data_dict['br']
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_fold, 'dw.edgelist_'+str(time_step)), 'w') as edge_file:
            for u, r_list in user_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[u])+' '+str(self.node2id[r])+'\n')
                    edge_file.write(str(self.node2id[r]) + ' ' + str(self.node2id[u]) + '\n')
            for b,r_list in business_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[b])+' '+str(self.node2id[r])+'\n')
                    edge_file.write(str(self.node2id[r])+' '+str(self.node2id[b])+'\n')
            for b, s_list in self.business_stars.items():
                for s in s_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[s]) + '\n')
                    edge_file.write(str(self.node2id[s]) + ' ' + str(self.node2id[b]) + '\n')

        # data for line
        print ('get data for line at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_fold, 'line.edgelist_'+str(time_step)), 'w') as edge_file:
            for u, r_list in user_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[u])+' '+str(self.node2id[r])+' '+str(1)+'\n')
                    edge_file.write(str(self.node2id[r])+' '+str(self.node2id[u])+' '+str(1)+'\n')
            for b,r_list in business_review.items():
                for r in r_list:
                    edge_file.write(str(self.node2id[b])+' '+str(self.node2id[r])+' '+str(1)+'\n')
                    edge_file.write(str(self.node2id[r])+' '+str(self.node2id[b])+' '+str(1)+'\n')
            for b, s_list in self.business_stars.items():
                for s in s_list:
                    edge_file.write(str(self.node2id[b]) + ' ' + str(self.node2id[s])+' '+str(1) + '\n')
                    edge_file.write(str(self.node2id[s]) + ' ' + str(self.node2id[b])+' '+str(1) + '\n')

        # data for esim
        print('get data for esim at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_fold, 'esim.metapath_'+str(time_step)), 'w') as metapath_file:
            metapath_file.write('bsb 1' + '\n')
            metapath_file.write('brurb 1'+'\n')
        with open(os.path.join(self.baseline_fold, 'esim.node_'+str(time_step)), 'w') as node_file:
            for a in self.user_list:
                node_file.write(a + ' u')
                node_file.write('\n')
            for p in self.review_list:
                node_file.write(p + ' r')
                node_file.write('\n')
            for c in self.business_list:
                node_file.write(c + ' b')
                node_file.write('\n')
            for t in self.stars_list:
                node_file.write(t + ' s')
                node_file.write('\n')
        with open(os.path.join(self.baseline_fold, 'esim.link_'+str(time_step)), 'w') as net_file:
            for a, p_list in user_review.items():
                for p in list(p_list):
                    net_file.write(a + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
                    net_file.write(p + ' ')
                    net_file.write(a + ' ')
                    net_file.write('\n')
            for p, c_list in business_review.items():
                for c in list(c_list):
                    net_file.write(p + ' ')
                    net_file.write(c + ' ')
                    net_file.write('\n')
                    net_file.write(c + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
            for p, t_list in self.business_stars.items():
                for t in list(t_list):
                    net_file.write(p + ' ')
                    net_file.write(t + ' ')
                    net_file.write('\n')
                    net_file.write(t + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')

    def get_data_4_dane(self,current_data_dict,time_step):
        # data for DANE
        print('get data for dane...')
        user_review = current_data_dict['ur']
        business_review = current_data_dict['br']
        adj_matrix = np.zeros([len(self.node2id), len(self.node2id)], dtype=float)
        for a, p_list in user_review.items():
            for p in p_list:
                adj_matrix[self.node2id[a]][self.node2id[p]] = 1
                adj_matrix[self.node2id[p]][self.node2id[a]] = 1
        for c, p_list in business_review.items():
            for p in p_list:
                adj_matrix[self.node2id[c]][self.node2id[p]] = 1
                adj_matrix[self.node2id[p]][self.node2id[c]] = 1
        for t, p_list in self.business_stars.items():
            for p in p_list:
                adj_matrix[self.node2id[t]][self.node2id[p]] = 1
                adj_matrix[self.node2id[p]][self.node2id[t]] = 1
        csr_mtx = csr_matrix(adj_matrix)
        scipy.io.savemat(os.path.join(self.baseline_fold, 'dane_csr_lp_' + str(time_step) + '.mat'),
                         {'yelpWWW_' + str(time_step): csr_mtx})

    def get_data_4_m2v(self, metapath, num_walks, walk_length,current_data_dict,time_step):
        """
        get data for metapath2vec
        over all authors!!
        :return:
        """
        user_review = current_data_dict['ur']
        review_user = current_data_dict['ru']
        business_review = current_data_dict['br']
        review_business = current_data_dict['rb']
        # data for metapath2vec
        print('get data for metapath2vec at time step: {}'.format(time_step))
        print ('generating paths randomly via {}...'.format(metapath))
        file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + \
                    str(walk_length) + '_paths_'+str(time_step)+'.txt'
        outfile = open(os.path.join(self.baseline_fold, file_name), 'w')
        if metapath == 'bsb':
            for business in self.business_stars:
                for j in xrange(0, num_walks):
                    outline = business
                    for i in xrange(0, walk_length):
                        stars_list = list(self.business_stars[business])
                        stars = random.choice(stars_list)
                        outline += ' ' + stars
                        business_list = list(self.stars_business[stars])
                        business = random.choice(business_list)
                        outline += ' ' + business
                    outfile.write(outline + "\n")
        elif metapath == 'brurb':
            for business in business_review:
                for j in xrange(0, num_walks):
                    outline = business
                    for i in xrange(0, walk_length):
                        if business_review.has_key(business):
                            review_list = list(business_review[business])
                            review = random.choice(review_list)
                        else:
                            review = random.choice(self.review_list)
                        outline += ' ' + review
                        if review_user.has_key(review):
                            user_list = list(review_user[review])
                            user = random.choice(user_list)
                        else:
                            user = random.choice(self.user_list)
                        if user_review.has_key(user):
                            review_list = list(user_review[user])
                            review = random.choice(review_list)
                        else:
                            review = random.choice(self.review_list)
                        if review_business.has_key(review):
                            business_list = list(review_business[review])
                            business = random.choice(business_list)
                        else:
                            business = random.choice(self.business_list)
                        outline += ' ' + business
                    outfile.write(outline + "\n")
        outfile.close()

    def get_data_4_our_symmetrical(self,current_data_dict,time_stem):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical) at time step: {}'.format(time_stem))
        ur_adj_mtx = current_data_dict['ur_adj_mtx']
        br_adj_mtx = current_data_dict['br_adj_mtx']

        bsb_adj_mtx = np.matmul(self.bs_adj_mtx,self.bs_adj_mtx.transpose())
        bsb_csr_mtx = csr_matrix(bsb_adj_mtx)
        self.save_mat(bsb_csr_mtx, 'bsb_csr_'+str(time_stem))

        bru_adj_mtx = np.matmul(br_adj_mtx,ur_adj_mtx.transpose())
        brurb_adj_mtx = np.matmul(bru_adj_mtx,bru_adj_mtx.transpose())
        brurb_csr_mtx = csr_matrix(brurb_adj_mtx)
        self.save_mat(brurb_csr_mtx, 'brurb_csr_'+str(time_stem))

    def save_mat(self, matrix, relation_name):
        """
        save data to mat
        :param matrix:
        :param relation_name:
        :return:
        """
        scipy.io.savemat(os.path.join(self.our_fold, relation_name),
                         {relation_name: matrix})


if __name__ == '__main__':
    dh = DataHelper(data_dir='../data/yelpWWW/oriData/',
                    baseline_dir='../baseline/yelpWWWDynamic_lp/',
                    our_dir='../data/yelpWWWDynamic_lp/')
    dh.load_data()

    dh.split_data(load_from_file=False)