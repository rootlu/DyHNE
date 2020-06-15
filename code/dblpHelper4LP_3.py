# coding: utf-8
# author: lu yf
# create date: 2018/8/9
# 每种类型边各隐藏0.2

import os
import numpy as np
import scipy.io
import random
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
        self.edge_list = []

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
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(author_name):
                self.node2id[author_name] = len(self.node2id)
            self.edge_list.append((paper_name,author_name))
        with open(os.path.join(self.dblp_data_fold, 'paper_conf.txt')) as pc_file:
            self.pc_lines = pc_file.readlines()
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.conf_list.append(token[1])
            paper_name = 'p' + token[0]
            conf_name = 'c' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(conf_name):
                self.node2id[conf_name] = len(self.node2id)
            self.edge_list.append((paper_name, conf_name))
        with open(os.path.join(self.dblp_data_fold, 'paper_term.txt')) as pt_file:
            self.pt_lines = pt_file.readlines()
        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.term_list.append(token[1])
            paper_name = 'p' + token[0]
            term_name = 't' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(term_name):
                self.node2id[term_name] = len(self.node2id)
            self.edge_list.append((paper_name,term_name))
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))

        self.save_node2id()
        print ('#paper:{}, #author:{}, #conf:{}, #term:{}, #node: {}, #edge: {}'.
               format(len(self.paper_list), len(self.author_list),
                      len(self.conf_list), len(self.term_list),
                      len(self.node2id),len(self.edge_list)))

    def save_node2id(self):
        with open(os.path.join(self.baseline_data_fold, 'node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

    def split_data(self,load_from_file,split_ratio):
        """
        hidden edges
        load history data from file or not
        :param load_from_file:
        :param split_ratio:
        :return:
        """
        if load_from_file:
            print('split data with history data...')
            pass
        else:
            print('split data again...')
            self.test_edge = random.sample(self.edge_list,int(len(self.edge_list) * split_ratio))
            self.train_edge = list(set(self.edge_list)-set(self.test_edge))

            self.train_pa_lines = []
            self.test_pa_lines = []
            self.train_pc_lines = []
            self.test_pc_lines = []
            self.train_pt_lines = []
            self.test_pt_lines = []
            for e in self.train_edge:
                if e[1][0] == 'a':
                    self.train_pa_lines.append(e[0][1:]+'\t'+e[1][1:])
                elif e[1][0] == 'c':
                    self.train_pc_lines.append(e[0][1:]+'\t'+e[1][1:])
                elif e[1][0] == 't':
                    self.train_pt_lines.append(e[0][1:]+'\t'+e[1][1:])
                else:
                    print('error train edge: {}'.format(e))
            for e in self.test_edge:
                if e[1][0] == 'a':
                    self.test_pa_lines.append(e[0][1:]+'\t'+e[1][1:])
                elif e[1][0] == 'c':
                    self.test_pc_lines.append(e[0][1:]+'\t'+e[1][1:])
                elif e[1][0] == 't':
                    self.test_pt_lines.append(e[0][1:]+'\t'+e[1][1:])
                else:
                    print('error test edge: {}'.format(e))

            # self.test_pa_lines = random.sample(self.pa_lines,int(len(self.pa_lines) * split_ratio))
            # self.train_pa_lines = list(set(self.pa_lines)-set(self.test_pa_lines))
            #
            # self.test_pc_lines = random.sample(self.pc_lines, int(len(self.pc_lines) * split_ratio))
            # self.train_pc_lines = list(set(self.pc_lines)-set(self.test_pc_lines))
            #
            # self.test_pt_lines = random.sample(self.pt_lines, int(len(self.pt_lines) * split_ratio))
            # self.train_pt_lines = list(set(self.pt_lines)-set(self.test_pt_lines))

            self.save_train_test_edge(self.train_pa_lines,'train_pa')
            self.save_train_test_edge(self.test_pa_lines,'test_pa')
            self.save_train_test_edge(self.train_pc_lines,'train_pc')
            self.save_train_test_edge(self.test_pa_lines,'test_pc')
            self.save_train_test_edge(self.train_pt_lines,'train_pt')
            self.save_train_test_edge(self.test_pt_lines,'test_pt')

        print('#train pa: {}, #test pa: {}'.format(len(self.train_pa_lines), len(self.test_pa_lines)))
        print('#train pc: {}, #test pc: {}'.format(len(self.train_pc_lines), len(self.test_pc_lines)))
        print('#train pt: {}, #test pt: {}'.format(len(self.train_pt_lines), len(self.test_pt_lines)))

    def save_train_test_edge(self,lines,edge_type):
        edge_file = open(os.path.join(self.our_data_fold, edge_type+'.txt'), 'w')
        for line in lines:
            edge_file.write(line)
        edge_file.close()

    def process_data(self):
        """
        process data for baselines and our
        get paper_author dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        print ('process data...')
        self.pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        self.train_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        self.test_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            self.pa_adj_mtx[row][col] = 1
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

        self.train_pc_adj_mtx = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.train_pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.train_pc_adj_mtx[row][col] = 1

            paper_name = 'p' + token[0]
            conf_name = 'c' + token[1]
            if not self.paper_conf.has_key(paper_name):
                self.paper_conf[paper_name] = set()
            self.paper_conf[paper_name].add(conf_name)
            if not self.conf_paper.has_key(conf_name):
                self.conf_paper[conf_name] = set()
            self.conf_paper[conf_name].add(paper_name)

        self.train_pt_adj_mtx = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in self.train_pt_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.train_pt_adj_mtx[row][col] = 1

            paper_name = 'p' + token[0]
            term_name = 't' + token[1]
            if not self.paper_term.has_key(paper_name):
                self.paper_term[paper_name] = set()
            self.paper_term[paper_name].add(term_name)
            if not self.term_paper.has_key(term_name):
                self.term_paper[term_name] = set()
            self.term_paper[term_name].add(paper_name)

        self.apa_adj_mtx = np.matmul(self.pa_adj_mtx.transpose(),self.pa_adj_mtx)
        self.train_apa_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(),self.train_pa_adj_mtx)
        self.test_apa_adj_mtx = np.matmul(self.test_pa_adj_mtx.transpose(),self.test_pa_adj_mtx)

    def get_aa_pos_neg_data(self,data_type,negative_aa):
        """
        get positive and negative co-author data for link prediction
        """
        print('get {} aa data...'.format(data_type))
        if data_type == 'train':
            apa_adj_mtx = self.train_apa_adj_mtx
        elif data_type == 'test':
            apa_adj_mtx = self.test_apa_adj_mtx
        # row, col = np.nonzero(apa_adj_mtx)
        row, col = np.nonzero(np.triu(apa_adj_mtx))
        print('#{} aa: {}'.format(data_type,len(row)))
        with open(os.path.join(self.our_data_fold,data_type+'_aa_pos.txt'), 'w') as aa_pos_f:
            for i in xrange(len(row)):
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                aa_pos_f.write(str(pos_a_1)+'\t'+str(pos_a_2)+'\n')

        neg_aa_list = random.sample(negative_aa,len(row))

        with open(os.path.join(self.our_data_fold,data_type+'_aa_neg.txt'), 'w') as aa_neg_f:
            for n_aa in neg_aa_list:
                aa_neg_f.write(str(n_aa[0]) + '\t' + str(n_aa[1]) + '\n')

    def get_data_4_baselines(self):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec ...')
        with open(os.path.join(self.baseline_data_fold, 'dw.edgelist_lp'), 'w') as edge_file:
            for a, p_list in self.author_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[a]) + ' ' + str(self.node2id[p]) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[a]) + '\n')
            for c, p_list in self.conf_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[p]) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[c]) + '\n')
            for t, p_list in self.term_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[t]) + ' ' + str(self.node2id[p]) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[t]) + '\n')

        print ('get data for line ...')
        with open(os.path.join(self.baseline_data_fold, 'line.edgelist_lp'), 'w') as edge_file:
            for a, p_list in self.author_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[a]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[a]) + ' ' + str(1) + '\n')
            for c, p_list in self.conf_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[c]) + ' ' + str(1) + '\n')
            for t, p_list in self.term_paper.items():
                for p in p_list:
                    edge_file.write(str(self.node2id[t]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
                    edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[t]) + ' ' + str(1) + '\n')

        # data for esim
        print('get data for esim...')
        with open(os.path.join(self.baseline_data_fold, 'esim.metapath_lp'), 'w') as metapath_file:
            metapath_file.write('apa 1' + '\n')
            metapath_file.write('apcpa 1'+'\n')
            metapath_file.write('aptpa 1')
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
        # for j in tqdm(xrange(0, num_walks)):  # wnum walks
        #     for author in self.author_list:
        #         outline = 'a' + str(author)
        #         author = 'a' + str(author)
        #         for i in xrange(walk_length):
        #             if metapath == 'apa':
        #                 # select 'p'
        #                 if self.author_paper.has_key(author):
        #                     next_p_list = list(self.author_paper[author])
        #                     next_p_node = random.choice(next_p_list)
        #                 else:
        #                     next_p_node = 'p' + random.choice(self.paper_list)
        #                 outline += ' ' + str(next_p_node)
        #                 # select 'a'
        #                 if self.paper_author.has_key(next_p_node):
        #                     next_a_list = list(self.paper_author[next_p_node])
        #                     next_a_node = random.choice(next_a_list)
        #                 else:
        #                     next_a_node = 'a' + random.choice(self.author_list)
        #                 outline += ' ' + str(next_a_node)
        #                 author = next_a_node
        #
        #             elif metapath == 'apcpa':
        #                 # select 'p'
        #                 if self.author_paper.has_key(author):
        #                     next_p_list = list(self.author_paper[author])
        #                     next_p_node = random.choice(next_p_list)
        #                 else:
        #                     next_p_node = 'p' + random.choice(self.paper_list)
        #                 outline += ' ' + str(next_p_node)
        #                 # select 'c'
        #                 next_c_list = list(self.paper_conf[next_p_node])
        #                 next_c_node = random.choice(next_c_list)
        #                 outline += ' ' + next_c_node
        #                 # select 'p'
        #                 next_p_list = list(self.conf_paper[next_c_node])
        #                 next_p_node = random.choice(next_p_list)
        #                 outline += ' ' + str(next_p_node)
        #                 # select 'a'
        #                 if self.paper_author.has_key(next_p_node):
        #                     next_a_list = list(self.paper_author[next_p_node])
        #                     next_a_node = random.choice(next_a_list)
        #                 else:
        #                     next_a_node = 'a' + random.choice(self.author_list)
        #                 outline += ' ' + str(next_a_node)
        #                 author = next_a_node
        #
        #             elif metapath == 'aptpa':
        #                 # select 'p'
        #                 if self.author_paper.has_key(author):
        #                     next_p_list = list(self.author_paper[author])
        #                     next_p_node = random.choice(next_p_list)
        #                 else:
        #                     next_p_node = 'p' + random.choice(self.paper_list)
        #                 outline += ' ' + str(next_p_node)
        #                 # select 't'
        #                 if self.paper_term.has_key(next_p_node):
        #                     next_t_list = list(self.paper_term[next_p_node])
        #                     next_t_node = random.choice(next_t_list)
        #                 else:
        #                     next_t_node = 't' + random.choice(self.term_list)
        #                 outline += ' ' + next_t_node
        #                 # select 'p'
        #                 next_p_list = list(self.term_paper[next_t_node])
        #                 next_p_node = random.choice(next_p_list)
        #                 outline += ' ' + str(next_p_node)
        #                 # select 'a'
        #                 if self.paper_author.has_key(next_p_node):
        #                     next_a_list = list(self.paper_author[next_p_node])
        #                     next_a_node = random.choice(next_a_list)
        #                 else:
        #                     next_a_node = 'a' + random.choice(self.author_list)
        #                 outline += ' ' + str(next_a_node)
        #                 author = next_a_node
        #         outfile.write(outline + "\n")
        if metapath == 'apa':
            for author in self.author_paper:
                for j in xrange(0, num_walks):
                    outline = author
                    for i in xrange(0, walk_length):
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            continue
                        outline += ' ' + next_p_node
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            continue
                        outline += ' ' + next_a_node
                        author = next_a_node
                    outfile.write(outline + "\n")
        elif metapath == 'apcpa':
            for author in self.author_paper:
                for j in xrange(0, num_walks):
                    outline = author
                    for i in xrange(0, walk_length):
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            continue
                        outline += ' ' + next_p_node
                        if self.paper_conf.has_key(next_p_node):
                            next_c_list = list(self.paper_conf[next_p_node])
                            next_c_node = random.choice(next_c_list)
                        else:
                            continue
                        outline += ' ' + next_c_node
                        if self.conf_paper.has_key(next_c_node):
                            next_p_list = list(self.conf_paper[next_c_node])
                            next_p_node = random.choice(next_p_list)
                        else:
                            continue
                        outline += ' ' + next_p_node
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            continue
                        outline += ' ' + next_a_node
                        author = next_a_node
                    outfile.write(outline + "\n")
        elif metapath == 'aptpa':
            for author in self.author_paper:
                for j in xrange(0, num_walks):
                    outline = author
                    for i in xrange(0, walk_length):
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                            next_p_node = random.choice(next_p_list)
                        else:
                            continue
                        outline += ' ' + next_p_node
                        if self.paper_term.has_key(next_p_node):
                            next_t_list = list(self.paper_term[next_p_node])
                            next_t_node = random.choice(next_t_list)
                        else:
                            continue
                        outline += ' ' + next_t_node
                        if self.term_paper.has_key(next_t_node):
                            next_p_list = list(self.term_paper[next_t_node])
                            next_p_node = random.choice(next_p_list)
                        else:
                            continue
                        outline += ' ' + next_p_node
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                            next_a_node = random.choice(next_a_list)
                        else:
                            continue
                        outline += ' ' + next_a_node
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

        apc_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(),self.train_pc_adj_mtx)
        apcpa_adj_mtx = np.matmul(apc_adj_mtx,apc_adj_mtx.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_lp')

        apt_adj_mtx = np.matmul(self.train_pa_adj_mtx.transpose(), self.train_pt_adj_mtx)
        aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_lp')

    def save_mat(self, matrix, relation_name):
        """
        save data to mat
        :param matrix:
        :param relation_name:
        :return:
        """
        scipy.io.savemat(os.path.join(self.our_data_fold, relation_name),
                         {relation_name: matrix})


if __name__ == '__main__':
    dh = DataHelper(data_dir='../data/dblp/oriData/',
                    baseline_dir='../baseline/dblp_lp_del_edges',
                    our_dir='../data/dblp_lp_del_edges')
    dh.load_data()

    dh.split_data(load_from_file=False,split_ratio=0.2)
    dh.process_data()

    # negative apa
    neg_aa = np.where(dh.apa_adj_mtx == 0)
    cand_neg_aa_list = zip(neg_aa[0], neg_aa[1])

    dh.get_data_4_our_symmetrical()

    dh.get_data_4_baselines()

    dh.get_aa_pos_neg_data('train',cand_neg_aa_list)
    dh.get_aa_pos_neg_data('test',cand_neg_aa_list)

    dh.get_data_4_m2v('apa', 10, 50)
    dh.get_data_4_m2v('apcpa', 10, 50)
    dh.get_data_4_m2v('aptpa', 10, 50)



