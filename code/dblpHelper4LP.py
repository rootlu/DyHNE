# coding:utf-8
# author: lu yf
# create date: 2018/7/13

import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix
from tqdm import tqdm


class DataHelper:
    def __init__(self,data_dir,baseline_dir,our_dir):
        self.dblp_data_fold = data_dir
        self.baseline_data_fold = baseline_dir
        self.our_data_fold = our_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []

        self.paper_author = {}
        self.author_paper = {}
        self.paper_conf = {}
        self.conf_paper = {}
        self.paper_term = {}
        self.term_paper = {}

        self.node2id = {}

        self.train_pa_lines = []
        self.test_pa_lines = []
        self.pa_lines = []
        self.pc_lines = []
        self.pt_lines = []
        self.train_pa_adj_mtx = np.zeros(0)
        self.test_pa_adj_mtx = np.zeros(0)
        self.pc_adj_mtx = np.zeros(0)
        self.pt_adj_mtx = np.zeros(0)

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print ('loading data...')
        with open(os.path.join(self.dblp_data_fold, 'paper_author.txt')) as pa_file:
            self.pa_lines = pa_file.readlines()
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.author_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_conf.txt')) as pc_file:
            self.pc_lines = pc_file.readlines()
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.conf_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_term.txt')) as pt_file:
            self.pt_lines = pt_file.readlines()
        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.term_list.append(token[1])
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))
        print ('#paper:{}, #author:{}, #conf:{}, term:{}'.format(len(self.paper_list), len(self.author_list),
                                                                 len(self.conf_list), len(self.term_list)))
        self.pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pa_adj_mtx[row][col] = 1

    def split_data(self,load_from_file):
        """
        hidden edges
        load history data from file or not
        :param load_from_file:
        :return:
        """
        if load_from_file:
            print('split data with history data...')
            with open('../data/dblp_lp/train_pa.txt','r') as tr_pa_f:
                self.train_pa_lines = tr_pa_f.readlines()
            with open('../data/dblp_lp/test_pa.txt','r') as te_pa_f:
                self.test_pa_lines = te_pa_f.readlines()
        else:
            print('split data again...')
            # self.train_pa_lines = random.sample(self.pa_lines, int(0.8 * len(self.pa_lines)))
            # self.test_pa_lines = list(set(self.pa_lines) - set(self.train_pa_lines))

            pa_adj_mtx = self.pa_adj_mtx
            target_num = 0.2*len(self.pa_lines)
            test_pa = set()
            while 1:
                # print('{}/{}'.format(len(test_pa),target_num))
                cand_pa = random.choice(self.pa_lines)
                tokens = cand_pa.split('\t')
                cand_p = tokens[0]
                cand_a = tokens[1]
                if sum(pa_adj_mtx[:,int(cand_a)]) > 1:  # at least two papers
                    test_pa.add(cand_pa)
                    pa_adj_mtx[int(cand_p)][int(cand_a)] = 0
                if len(test_pa) >= target_num:
                    break
            self.test_pa_lines = test_pa
            self.train_pa_lines = list(set(self.pa_lines) - set(self.test_pa_lines))
        print('#train pa: {}, #test pa: {}'.format(len(self.train_pa_lines), len(self.test_pa_lines)))

    def process_data(self):
        """
        process data for baselines and our
        get paper_author dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        print ('process data...')
        self.train_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        self.test_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(author_name):
                self.node2id[author_name] = len(self.node2id)

            if line in self.train_pa_lines:
                self.train_pa_adj_mtx[row][col] = 1
                if not self.paper_author.has_key(paper_name):
                    self.paper_author[paper_name] = set()
                self.paper_author[paper_name].add(author_name)
                if not self.author_paper.has_key(author_name):
                    self.author_paper[author_name] = set()
                self.author_paper[author_name].add(paper_name)
            elif line in self.test_pa_lines:
                self.test_pa_adj_mtx[row][col] = 1

        self.pc_adj_mtx = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pc_adj_mtx[row][col] = 1

            paper_name = 'p' + token[0]
            conf_name = 'c' + token[1]
            if not self.paper_conf.has_key(paper_name):
                self.paper_conf[paper_name] = set()
            self.paper_conf[paper_name].add(conf_name)
            if not self.conf_paper.has_key(conf_name):
                self.conf_paper[conf_name] = set()
            self.conf_paper[conf_name].add(paper_name)
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(conf_name):
                self.node2id[conf_name] = len(self.node2id)

        self.pt_adj_mtx = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pt_adj_mtx[row][col] = 1

            paper_name = 'p' + token[0]
            term_name = 't' + token[1]
            if not self.paper_term.has_key(paper_name):
                self.paper_term[paper_name] = set()
            self.paper_term[paper_name].add(term_name)
            if not self.term_paper.has_key(term_name):
                self.term_paper[term_name] = set()
            self.term_paper[term_name].add(paper_name)
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(term_name):
                self.node2id[term_name] = len(self.node2id)

        self.apa_adj_mtx = np.matmul(self.pa_adj_mtx.transpose(),self.pa_adj_mtx)
        self.train_apa_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(),self.train_pa_adj_mtx)
        self.test_apa_adj_mtx = np.matmul(self.test_pa_adj_mtx.transpose(),self.test_pa_adj_mtx)

    def save_train_test_pa_data(self):
        """
        save paper-author in train set and test set
        save author co-appear in train and test
        :return:
        """
        with open(os.path.join(self.our_data_fold,'train_pa.txt'),'w') as train_pa_f:
            for line in self.train_pa_lines:
                train_pa_f.write(line)
        with open(os.path.join(self.our_data_fold,'test_pa.txt'),'w') as test_pa_f:
            for line in self.test_pa_lines:
                test_pa_f.write(line)

        print('get good test pa...')
        test_a = map(lambda x: x.strip().split('\t')[1], self.test_pa_lines)
        train_a = map(lambda x: x.strip().split('\t')[1], self.train_pa_lines)

        good_a = list(set(train_a) & set(test_a))
        self.good_author = good_a

        with open(os.path.join(self.our_data_fold,'good_author_4_test.txt'), 'w') as t_pa_f:
            for a in good_a:
                t_pa_f.write(str(a) + '\n')

    def get_pa_pos_neg_data(self,data_type):
        print('get {} pa data...'.format(data_type))
        if data_type == 'train':
            pa_adj_mtx = self.train_pa_adj_mtx
        elif data_type == 'test':
            pa_adj_mtx = self.test_pa_adj_mtx
        row, col = np.nonzero(pa_adj_mtx)
        print('#{} pa: {}'.format(data_type, len(row)))
        with open(os.path.join(self.our_data_fold, data_type + '_pa_pos.txt'), 'w') as aa_pos_f:
            for i in xrange(len(row)):
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                aa_pos_f.write(str(pos_a_1) + '\t' + str(pos_a_2) + '\n')

        # negative pa
        zero_pa = np.where(self.pa_adj_mtx == 0)
        cand_pa_list = zip(zero_pa[0], zero_pa[1])
        neg_pa_list = random.sample(cand_pa_list, len(row))

        with open(os.path.join(self.our_data_fold, data_type + '_pa_neg.txt'), 'w') as aa_neg_f:
            for neg_pa in neg_pa_list:
                aa_neg_f.write(str(neg_pa[0]) + '\t' + str(neg_pa[1]) + '\n')

    def get_aa_pos_neg_data(self,data_type):
        """
        get positive and negative co-author data for link prediction
        :param data_type: train or test
        :return:
        """
        print('get {} aa data...'.format(data_type))
        if data_type == 'train':
            apa_adj_mtx = self.train_apa_adj_mtx
        elif data_type == 'test':
            apa_adj_mtx = self.test_apa_adj_mtx
        # triu_apa_adj_mtx = np.triu(apa_adj_mtx)  # a1-a2 == a2-a1
        row, col = np.nonzero(apa_adj_mtx)
        print('#{} aa: {}'.format(data_type,len(row)))
        with open(os.path.join(self.our_data_fold,data_type+'_aa_pos.txt'), 'w') as aa_pos_f:
            for i in xrange(len(row)):
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                aa_pos_f.write(str(pos_a_1)+'\t'+str(pos_a_2)+'\n')

        # negative apa
        zero_aa = np.where(self.apa_adj_mtx == 0)
        cand_aa_list = zip(zero_aa[0],zero_aa[1])
        neg_aa_list = random.sample(cand_aa_list,len(row))

        with open(os.path.join(self.our_data_fold,data_type+'_aa_neg.txt'), 'w') as aa_neg_f:
            for neg_aa in neg_aa_list:
                aa_neg_f.write(str(neg_aa[0]) + '\t' + str(neg_aa[1]) + '\n')

    def get_data_4_baselines(self):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec ...')
        with open(os.path.join(self.baseline_data_fold, 'dw.adjlist_lp'), 'w') as adj_file:
            for node_name, node_id in self.node2id.items():
                adj_line = str(node_id)
                if node_name[0] == 'a':
                    if self.author_paper.has_key(node_name):
                        next_node_list = list(self.author_paper[node_name])
                    else:
                        adj_file.write(adj_line + '\n')
                        print('iso!!!')
                        continue
                        # next_node_list = map(lambda x: 'p' + x, self.paper_list)
                elif node_name[0] == 'p':
                    if self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name]) + list(
                            self.paper_term[node_name])
                    elif self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_term[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_conf[node_name])
                elif node_name[0] == 'c':
                    next_node_list = list(self.conf_paper[node_name])
                elif node_name[0] == 't':
                    next_node_list = list(self.term_paper[node_name])

                for n_n_name in next_node_list:
                    n_n_id = self.node2id[n_n_name]
                    adj_line += ' ' + str(n_n_id)
                adj_file.write(adj_line + '\n')

        with open(os.path.join(self.baseline_data_fold, 'dw.node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for line
        print ('get data for line ...')
        with open(os.path.join(self.baseline_data_fold, 'line.edgelist_lp'), 'w') as edge_file:
            for node_name, node_id in self.node2id.items():
                if node_name[0] == 'a':
                    if self.author_paper.has_key(node_name):
                        next_node_list = list(self.author_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        print('iso!!!')
                        continue
                        # next_node_list = map(lambda x: 'p' + x, self.paper_list)
                elif node_name[0] == 'p':
                    if self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name]) + list(
                            self.paper_term[node_name])
                    elif self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_term[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_conf[node_name])
                elif node_name[0] == 'c':
                    next_node_list = list(self.conf_paper[node_name])
                elif node_name[0] == 't':
                    next_node_list = list(self.term_paper[node_name])

                for n_n_name in next_node_list:
                    n_n_id = self.node2id[n_n_name]
                    edge_line = str(node_id) + ' ' + str(n_n_id) + ' ' + str(1)
                    edge_file.write(edge_line + '\n')

        with open(os.path.join(self.baseline_data_fold, 'line.node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for esim
        print('get data for esim...')
        with open(os.path.join(self.baseline_data_fold, 'esim.metapath_lp'), 'w') as metapath_file:
            metapath_file.write('apa 0.1' + '\n')
            metapath_file.write('apcpa 0.7'+'\n')
            metapath_file.write('aptpa 0.2')
            metapath_file.write('\n')
        with open(os.path.join(self.baseline_data_fold, 'esim.node_lp'), 'w') as node_file:
            for a in self.author_list:
                node_file.write('a' + a + ' a')
                node_file.write('\n')
            for p in self.paper_list:
                node_file.write('p' + p + ' p')
                node_file.write('\n')
            for c in self.conf_list:
                node_file.write('c' + c + ' c')
                node_file.write('\n')
            for t in self.term_list:
                node_file.write('t' + t + ' t')
                node_file.write('\n')
        with open(os.path.join(self.baseline_data_fold, 'esim.link_lp'), 'w') as net_file:
            for a, p_list in self.author_paper.items():
                for p in list(p_list):
                    net_file.write(a + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
                    net_file.write(p + ' ')
                    net_file.write(a + ' ')
                    net_file.write('\n')
            for p, c_list in self.paper_conf.items():
                for c in list(c_list):
                    net_file.write(p + ' ')
                    net_file.write(c + ' ')
                    net_file.write('\n')
                    net_file.write(c + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
            for p, t_list in self.paper_term.items():
                for t in list(t_list):
                    net_file.write(p + ' ')
                    net_file.write(t + ' ')
                    net_file.write('\n')
                    net_file.write(t + ' ')
                    net_file.write(p + ' ')
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
        outfile = open(os.path.join(self.baseline_data_fold, file_name), 'w')
        for j in tqdm(xrange(0, num_walks)):  # wnum walks
            for author in self.author_list:
                outline = 'a' + str(author)
                author = 'a' + str(author)
                for i in xrange(walk_length):
                    if metapath == 'apa':
                        # select 'p'
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            next_p_node = 'p' + random.choice(self.paper_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            next_a_node = 'a' + random.choice(self.author_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'apcpa':
                        # select 'p'
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            next_p_node = 'p' + random.choice(self.paper_list)
                        outline += ' ' + str(next_p_node)
                        # select 'c'
                        next_c_list = list(self.paper_conf[next_p_node])
                        next_c_node = random.choice(next_c_list)
                        outline += ' ' + next_c_node
                        # select 'p'
                        next_p_list = list(self.conf_paper[next_c_node])
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            next_a_node = 'a' + random.choice(self.author_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'aptpa':
                        # select 'p'
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            next_p_node = 'p' + random.choice(self.paper_list)
                        outline += ' ' + str(next_p_node)
                        # select 't'
                        if self.paper_term.has_key(next_p_node):
                            next_t_list = list(self.paper_term[next_p_node])
                            next_t_node = random.choice(next_t_list)
                        else:
                            next_t_node = 't' + random.choice(self.term_list)
                        outline += ' ' + next_t_node
                        # select 'p'
                        next_p_list = list(self.term_paper[next_t_node])
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            next_a_node = 'a' + random.choice(self.author_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node
                outfile.write(outline + "\n")

        outfile.close()

    def get_data_4_our_symmetrical(self):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical)...')

        apa_csr_mtx = csr_matrix(self.train_apa_adj_mtx)
        self.save_mat(apa_csr_mtx, 'apa_csr_lp')

        apc_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(),self.pc_adj_mtx)
        apcpa_adj_mtx = np.matmul(apc_adj_mtx,apc_adj_mtx.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_lp')

        apt_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(), self.pt_adj_mtx)
        aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_lp')

    def get_data_4_our_asymmetric(self):
        """
        get data for our model
        deal with asy. metapaths.
        :return:
        """
        print('get data for our (asymmetric)...')
        train_p = map(lambda x: int(x.strip().split('\t')[0]) + len(self.author_list), self.train_pa_lines)
        train_a = map(lambda x: int(x.strip().split('\t')[1]), self.train_pa_lines)
        row = np.array(train_a + train_p)
        col = np.array(train_p + train_a)
        data = np.ones(2 * len(self.train_pa_lines))
        ap_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.author_list) + len(self.paper_list),
                                       len(self.author_list) + len(self.paper_list)))
        self.save_mat(ap_csr_mtx, 'ap_csr_lp')
        self.save_mat(ap_csr_mtx.transpose(), 'pa_csr_lp')

        train_p = map(lambda x: int(x.strip().split('\t')[0]), self.pc_lines)
        train_c = map(lambda x: int(x.strip().split('\t')[1]) + len(self.paper_list), self.pc_lines)
        row = np.array(train_p + train_c)
        col = np.array(train_c + train_p)
        data = np.ones(2 * len(self.pc_lines))
        pc_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.conf_list),
                                       len(self.paper_list) + len(self.conf_list)))
        self.save_mat(pc_csr_mtx, 'pc_csr_lp')

        train_p = map(lambda x: int(x.strip().split('\t')[0]), self.pt_lines)
        train_t = map(lambda x: int(x.strip().split('\t')[1]) + len(self.paper_list), self.pt_lines)
        row = np.array(train_p + train_t)
        col = np.array(train_t + train_p)
        data = np.ones(2 * len(self.pt_lines))
        pt_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.term_list),
                                       len(self.paper_list) + len(self.term_list)))
        self.save_mat(pt_csr_mtx, 'pt_csr_lp')

        pc_dict = {}
        for pc in self.pc_lines:
            tokens = pc.strip().split('\t')
            pc_dict[tokens[0]] = tokens[1]  # 1 vs 1
        ac_dict = {}
        for pa in self.train_pa_lines:
            tokens = pa.strip().split('\t')
            if not ac_dict.has_key(tokens[1]):
                ac_dict[tokens[1]] = []
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
        self.save_mat(apc_csr_mtx, 'apc_csr_lp')

        pt_dict = {}
        for pt in self.pt_lines:
            tokens = pt.strip().split('\t')
            if not pt_dict.has_key(tokens[0]):
                pt_dict[tokens[0]] = []
            pt_dict[tokens[0]].append(tokens[1])
        at_dict = {}
        for pa in self.train_pa_lines:
            tokens = pa.strip().split('\t')
            if not at_dict.has_key(tokens[1]):
                at_dict[tokens[1]] = []
            for t in pt_dict[tokens[0]]:
                at_dict[tokens[1]].append(t)
        row = []
        col = []
        data = []
        for a, t_list in at_dict.items():
            at_weight = collections.Counter(t_list)
            for t in list(set(t_list)):
                row.append(int(a))
                col.append(int(t) + len(self.author_list))
                data.append(float(at_weight[t]))
        apt_csr_mtx = csr_matrix((data, (row, col)),
                                 shape=(len(self.term_list) + len(self.author_list),
                                        len(self.term_list) + len(self.author_list)))
        self.save_mat(apt_csr_mtx, 'apt_csr_lp')

    def save_mat(self, matrix, relation_name):
        """
        save data to mat
        :param matrix:
        :param relation_name:
        :return:
        """
        scipy.io.savemat(os.path.join(self.our_data_fold, relation_name),
                         {relation_name: matrix})

    def second_4_dw(self):
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec ...')
        with open(os.path.join('../baseline/dblp_lp/', 'dw.adjlist_lp'), 'w') as adj_file:
            for node_name, node_id in tqdm(self.node2id.items()):
                adj_line = str(node_id)
                if node_name[0] == 'a':
                    if self.author_paper.has_key(node_name):
                        next_node_list = list(self.author_paper[node_name])
                    # else:
                        # adj_file.write(adj_line + '\n')
                        # continue
                        # next_node_list = map(lambda x: 'p' + x, self.paper_list)
                elif node_name[0] == 'p':
                    if self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name]) + list(
                            self.paper_term[node_name])
                    elif self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_term[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_author[node_name]) + list(self.paper_conf[node_name])
                    elif not self.paper_term.has_key(node_name) and not self.paper_author.has_key(node_name):
                        next_node_list = list(self.paper_conf[node_name])
                elif node_name[0] == 'c':
                    next_node_list = list(self.conf_paper[node_name])
                elif node_name[0] == 't':
                    next_node_list = list(self.term_paper[node_name])

                for n_n_name in next_node_list:
                    n_n_id = self.node2id[n_n_name]
                    adj_line += ' ' + str(n_n_id)
                adj_file.write(adj_line + '\n')

        with open(os.path.join('../baseline/dblp_lp/', 'dw.node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

    def judge_isolated_nodes(self):
        author_papers = sum(self.pa_adj_mtx)  # paper number of a author
        author_more_2_papers = np.where(author_papers >= 2)[0]  # the author who write at least two papers
        num_can_hide_authors = np.sum(author_papers[author_more_2_papers])-len(author_more_2_papers)
        if num_can_hide_authors >= 0.2*len(self.pa_lines):
            print('no isolated nodes!')
        else:
            print('exist isolated nodes!')
            exit()


if __name__ == '__main__':
    dh = DataHelper(data_dir='../data/dblp/oriData/',
                    baseline_dir='../baseline/dblp_lp_no_iso',
                    our_dir='../data/dblp_lp_no_iso')
    dh.load_data()
    # dh.judge_isolated_nodes()

    dh.split_data(load_from_file=False)
    dh.process_data()
    # # dh.second_4_dw()

    dh.get_data_4_our_symmetrical()
    # dh.get_data_4_our_asymmetric()

    dh.get_data_4_baselines()

    dh.get_aa_pos_neg_data(data_type='train')
    dh.get_aa_pos_neg_data(data_type='test')

    dh.get_data_4_m2v('apa', 10, 50)
    dh.get_data_4_m2v('apcpa', 10, 50)
    dh.get_data_4_m2v('aptpa', 10, 50)
    # dh.get_data_4_m2v('apa',10,100)
    # dh.get_data_4_m2v('apcpa',10,100)
    # dh.get_data_4_m2v('aptpa',10,100)
    dh.save_train_test_pa_data()

    # # link prediction
    # dh.get_pa_pos_neg_data(data_type='train')
    # dh.get_pa_pos_neg_data(data_type='test')


