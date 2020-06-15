# coding:utf-8
# author: lu yf
# create date: 2018/7/17


import os
import numpy as np
import scipy.io
from scipy import sparse
import random
import collections
from scipy.sparse import csr_matrix
from tqdm import tqdm


class DataHelper:
    def __init__(self,data_dir):
        self.dblp_data_fold = data_dir
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

    def split_data_apa(self):
        print ('split data...')
        pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1

        apa_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pa_adj_matrix)
        triu_apa_adj_matrix = (np.triu(apa_adj_matrix)>=1).astype(int)
        all_apa = np.nonzero(triu_apa_adj_matrix)
        print('#all co-author: {}'.format(len(all_apa[0])))
        zip_apa = zip(all_apa[0], all_apa[1])
        test_apa = random.sample(zip_apa,int(0.2*len(all_apa[0])))
        # test_apa = set()
        # while 1:
        #     can_apa = random.choice(zip_apa)
        #     if sum(triu_apa_adj_matrix[:,can_apa[0]]) < 3 or sum(triu_apa_adj_matrix[:,can_apa[1]]) < 3:
        #         # 删除ap后成为孤立点
        #         continue
        #     else:
        #         test_apa.add(can_apa)
        #         triu_apa_adj_matrix[can_apa[0]][can_apa[1]] = 0
        #     if len(test_apa) >= int(0.2*len(all_apa[0])):
        #         break
        print('#test co-author: {}'.format(len(test_apa)))
        self.test_pa = []
        for apa in list(test_apa):
            a1_p_list = np.where(pa_adj_matrix[:,apa[0]] > 0)[0]
            a2_p_list = np.where(pa_adj_matrix[:,apa[1]] > 0)[0]
            shared_p = list(set(a1_p_list) & set(a2_p_list))
            for p in shared_p:
                self.test_pa.append(str(p)+'\t'+str(apa[0])+'\n')
                self.test_pa.append(str(p)+'\t'+str(apa[1])+'\n')
        self.test_pa = list(set(self.test_pa))
        print('#test paper-author: {}'.format(len(self.test_pa)))

        for line in tqdm(self.pa_lines):
            token = line.strip('\n').split('\t')
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(author_name):
                self.node2id[author_name] = len(self.node2id)

            if line in self.test_pa:
                pa_adj_matrix[int(token[0])][int(token[1])] = 0
                continue
            if not self.paper_author.has_key(paper_name):
                self.paper_author[paper_name] = set()
            self.paper_author[paper_name].add(author_name)
            if not self.author_paper.has_key(author_name):
                self.author_paper[author_name] = set()
            self.author_paper[author_name].add(paper_name)

        pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in tqdm(self.pc_lines):
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1

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

        pt_adj_matrix = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in tqdm(self.pt_lines):
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pt_adj_matrix[row][col] = 1

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

        self.apa_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pa_adj_matrix)
        apc_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pc_adj_matrix)
        self.apcpa_adj_matrix = np.matmul(apc_adj_matrix,apc_adj_matrix.transpose())
        apt_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pt_adj_matrix)
        self.aptpa_adj_matrix = np.matmul(apt_adj_matrix,apt_adj_matrix.transpose())

    def split_data_apc(self):
        print ('split data with...')
        pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1
        pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1

        apc_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pc_adj_matrix)
        all_apc = np.nonzero(apc_adj_matrix)
        test_apc = random.sample(all_apc,0.2*len(all_apc))
        self.test_pa = []
        self.test_pc = []
        for apc in test_apc:
            a_p_list = np.where(pa_adj_matrix[:,apc[0]] > 0)[0]
            c_p_list = np.where(pc_adj_matrix[:,apc[1]] > 0)[0]
            shared_p = list(set(a_p_list) & set(c_p_list))
            for p in shared_p:
                self.test_pa.append(p+'\t'+apc[0])
                self.test_pc.append(p+'\t'+apc[1])

        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(author_name):
                self.node2id[author_name] = len(self.node2id)

            if line in self.test_pa:
                continue
            if not self.paper_author.has_key(paper_name):
                self.paper_author[paper_name] = set()
            self.paper_author[paper_name].add(author_name)
            if not self.author_paper.has_key(author_name):
                self.author_paper[author_name] = set()
            self.author_paper[author_name].add(paper_name)

        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            paper_name = 'p' + token[0]
            conf_name = 'c' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(conf_name):
                self.node2id[conf_name] = len(self.node2id)
            if line in self.test_pc:
                continue
            if not self.paper_conf.has_key(paper_name):
                self.paper_conf[paper_name] = set()
            self.paper_conf[paper_name].add(conf_name)
            if not self.conf_paper.has_key(conf_name):
                self.conf_paper[conf_name] = set()
            self.conf_paper[conf_name].add(paper_name)

        for line in self.pt_lines:
            token = line.strip('\n').split('\t')
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

    def get_data_4_baselines(self):
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec ...')
        with open(os.path.join('../baseline/dblp_lp/', 'dw.adjlist_lp'), 'w') as adj_file:
            for node_name, node_id in tqdm(self.node2id.items()):
                adj_line = str(node_id)
                if node_name[0] == 'a':
                    if self.author_paper.has_key(node_name):
                        next_node_list = list(self.author_paper[node_name])
                    else:
                        continue
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

        # data for line
        print ('get data for line ...')
        with open(os.path.join('../baseline/dblp_lp/', 'line.edgelist_lp'), 'w') as edge_file:
            for node_name,node_id in tqdm(self.node2id.items()):
                if node_name[0] == 'a':
                    if self.author_paper.has_key(node_name):
                        next_node_list = list(self.author_paper[node_name])
                    else:
                        continue
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

        with open(os.path.join('../baseline/dblp_lp/', 'line.node2id_lp'), 'w') as n2id_file:
            for n_name,n_id in self.node2id.items():
                n2id_file.write(n_name+'\t'+str(n_id)+'\n')

        # data for esim
        print('get data for esim...')
        with open(os.path.join('../baseline/dblp_lp/', 'esim.metapath_lp'), 'w') as metapath_file:
            metapath_file.write('apa 0.1'+'\n')
            metapath_file.write('apcpa 0.7'+'\n')
            metapath_file.write('aptpa 0.2')
            metapath_file.write('\n')
        with open(os.path.join('../baseline/dblp_lp/', 'esim.node_lp'), 'w') as node_file:
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
        with open(os.path.join('../baseline/dblp_lp/','esim.link_lp'),'w') as net_file:
            for a,p_list in self.author_paper.items():
                for p in list(p_list):
                    net_file.write(a + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
                    net_file.write(p + ' ')
                    net_file.write(a + ' ')
                    net_file.write('\n')
            for p,c_list in self.paper_conf.items():
                for c in list(c_list):
                    net_file.write(p + ' ')
                    net_file.write(c + ' ')
                    net_file.write('\n')
                    net_file.write(c + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
            for p,t_list in self.paper_term.items():
                for t in list(t_list):
                    net_file.write(p + ' ')
                    net_file.write(t + ' ')
                    net_file.write('\n')
                    net_file.write(t + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')

    def get_data_4_m2v(self,metapath, num_walks, walk_length):
        # data for metapath2vec
        print('get data for metapath2vec')
        print ('generating paths randomly via {}...'.format(metapath))
        file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + str(walk_length) + '_paths_lp.txt'
        outfile = open(os.path.join('../baseline/dblp_lp/', file_name), 'w')
        for j in tqdm(xrange(0, num_walks)):  # wnum walks
            for author in self.author_list:
                outline = 'a'+str(author)
                author = 'a'+str(author)
                for i in xrange(walk_length):
                    if metapath == 'apa':
                        # select 'p'
                        # next_p_list = list(self.author_paper[author])
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                        else:
                            next_p_list = map(lambda x:'p'+x,self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        # next_a_list = list(self.paper_author[next_p_node])
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                        else:
                            next_a_list = map(lambda x:'a'+x,self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'apcpa':
                        # select 'p'
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
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
                        else:
                            next_a_list = map(lambda x: 'a' + x, self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'aptpa':
                        # select 'p'
                        if self.author_paper.has_key(author):
                            next_p_list = list(self.author_paper[author])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 't'
                        if self.paper_term.has_key(next_p_node):
                            next_t_list = list(self.paper_term[next_p_node])
                        else:
                            next_t_list = map(lambda x: 't' + x, self.term_list)
                        next_t_node = random.choice(next_t_list)
                        outline += ' ' + next_t_node
                        # select 'p'
                        next_p_list = list(self.term_paper[next_t_node])
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if self.paper_author.has_key(next_p_node):
                            next_a_list = list(self.paper_author[next_p_node])
                        else:
                            next_a_list = map(lambda x: 'a' + x, self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                outfile.write(outline + "\n")

        outfile.close()

    def get_data_4_our_symmetrical(self):
        print('get data for our (symmetrical)...')
        apa_csr_mtx = sparse.csr_matrix(self.apa_adj_matrix)
        self.save_mat(apa_csr_mtx,'apa_csr_lp')

        apcpa_csr_mtx = sparse.csr_matrix(self.apcpa_adj_matrix)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_lp')

        aptpa_csr_mtx = sparse.csr_matrix(self.aptpa_adj_matrix)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_lp')

    def get_data_4_our_asymmetric(self):
        print('get data for our (asymmetric)...')
        train_pa = list(set(self.pa_lines)-set(self.test_pa))
        train_p = map(lambda x:int(x.strip().split('\t')[0])+len(self.author_list),train_pa)
        train_a = map(lambda x:int(x.strip().split('\t')[1]),train_pa)
        row = np.array(train_a+train_p)
        col = np.array(train_p+train_a)
        data = np.ones(2*len(train_pa))
        ap_csr_mtx = csr_matrix((data, (row, col)),
                            shape=(len(self.author_list)+len(self.paper_list), len(self.author_list)+len(self.paper_list)))
        self.save_mat(ap_csr_mtx, 'ap_csr_lp')
        self.save_mat(ap_csr_mtx.transpose(),'pa_csr_lp')

        train_p = map(lambda x: int(x.strip().split('\t')[0]), self.pc_lines)
        train_c = map(lambda x: int(x.strip().split('\t')[1])+len(self.paper_list), self.pc_lines)
        row = np.array(train_p + train_c)
        col = np.array(train_c + train_p)
        data = np.ones(2*len(self.pc_lines))
        pc_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.conf_list),
                                       len(self.paper_list) + len(self.conf_list)))
        self.save_mat(pc_csr_mtx, 'pc_csr_lp')

        train_p = map(lambda x: int(x.strip().split('\t')[0]), self.pt_lines)
        train_t = map(lambda x: int(x.strip().split('\t')[1])+len(self.paper_list), self.pt_lines)
        row = np.array(train_p + train_t)
        col = np.array(train_t + train_p)
        data = np.ones(2*len(self.pt_lines))
        pt_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.paper_list) + len(self.term_list),
                                       len(self.paper_list) + len(self.term_list)))
        self.save_mat(pt_csr_mtx, 'pt_csr_lp')

        pc_dict = {}
        for pc in self.pc_lines:
            tokens = pc.strip().split('\t')
            pc_dict[tokens[0]] = tokens[1]  # 1 vs 1
        ac_dict = {}
        for pa in train_pa:
            tokens = pa.strip().split('\t')
            if not ac_dict.has_key(tokens[1]):
                ac_dict[tokens[1]] = []
            ac_dict[tokens[1]].append(pc_dict[tokens[0]])
        row = []
        col = []
        data = []
        for a,c_list in ac_dict.items():
            ac_weight = collections.Counter(c_list)
            for c in list(set(c_list)):
                row.append(int(a))
                col.append(int(c)+len(self.author_list))
                data.append(float(ac_weight[c]))
        apc_csr_mtx = csr_matrix((data, (row, col)),
                                shape=(len(self.conf_list) + len(self.author_list),
                                       len(self.conf_list) + len(self.author_list)))
        self.save_mat(apc_csr_mtx, 'apc_csr_lp')

        pt_dict = {}
        for pc in self.pc_lines:
            tokens = pc.strip().split('\t')
            if not pt_dict.has_key(tokens[0]):
                pt_dict[tokens[0]] = []
            pt_dict[tokens[0]].append(tokens[1])
        at_dict = {}
        for pa in train_pa:
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

    def save_mat(self,matrix,relation_name):
        scipy.io.savemat(os.path.join('../data/dblp_lp/',relation_name),
                         {relation_name:matrix})

    def get_test_aa_data(self):
        print('get test aa data ...')
        test_pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.test_pa:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            test_pa_adj_matrix[row][col] = 1

        pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1
        apa_adj_matrix = np.matmul(pa_adj_matrix.transpose(),pa_adj_matrix)

        test_aa_adj_matrix = np.matmul(test_pa_adj_matrix.transpose(),test_pa_adj_matrix)
        triu_apa_adj_matrix = np.triu(test_aa_adj_matrix)
        row,col = np.nonzero(triu_apa_adj_matrix)
        with open('../data/dblp_lp/test_aa_pos_neg.txt','w') as t_aa_p_n_f:
            for i in tqdm(xrange(len(row))):
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                neg_a_list = list(np.where(apa_adj_matrix[pos_a_1, :] == 0)[0])
                while 1:
                    neg_a = random.choice(neg_a_list)
                    if neg_a in self.good_author:
                        break
                t_aa_p_n_f.write(str(pos_a_1) + '\t' + str(pos_a_2) + '\t' + str(neg_a) + '\n')

    def get_train_aa_data(self):
        print('get train aa data...')
        train_pa = list(set(self.pa_lines) - set(self.test_pa))
        train_pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in train_pa:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            train_pa_adj_matrix[row][col] = 1

        pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1
        apa_adj_matrix = np.matmul(pa_adj_matrix.transpose(), pa_adj_matrix)

        train_aa_adj_matrix = np.matmul(train_pa_adj_matrix.transpose(), train_pa_adj_matrix)
        triu_apa_adj_matrix = np.triu(train_aa_adj_matrix)
        row, col = np.nonzero(triu_apa_adj_matrix)
        with open('../data/dblp_lp/train_aa_pos_neg.txt', 'w') as t_aa_p_n_f:
            for i in tqdm(xrange(len(row))):
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                neg_a_list = list(np.where(apa_adj_matrix[pos_a_1, :] == 0)[0])
                while 1:
                    neg_a = random.choice(neg_a_list)
                    if neg_a in self.good_author:
                        break
                t_aa_p_n_f.write(str(pos_a_1) + '\t' + str(pos_a_2) + '\t' + str(neg_a) + '\n')

    def get_test_ac_data(self,neg_num):
        print('get test ac data ...')
        with open('../data/dblp/oriData/good_test_pa.txt', 'r') as t_pa_f:
            test_pa = t_pa_f.readlines()
        test_pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in test_pa:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            test_pa_adj_matrix[row][col] = 1
        pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1
        test_ac_adj_matrix = np.matmul(test_pa_adj_matrix.transpose(),pc_adj_matrix)
        row,col = np.nonzero(test_ac_adj_matrix)
        with open('../data/dblp/oriData/test_ac_pos_neg.txt','w') as t_aa_p_n_f:
            for i in xrange(len(row)):
                pos_a = row[i]
                pos_c = col[i]
                neg_c_list = list(np.where(test_ac_adj_matrix[pos_a, :] == 0)[0])
                neg_c_k_list = random.sample(neg_c_list, neg_num)
                neg_c = ' '.join(map(lambda x: str(x), neg_c_k_list))
                t_aa_p_n_f.write(str(pos_a) + '\t' + str(pos_c) + '\t' + neg_c + '\n')

    def get_train_ac_data(self,neg_num):
        print('get train ac data...')
        with open('../data/dblp/oriData/good_test_pa.txt', 'r') as t_pa_f:
            test_pa = t_pa_f.readlines()
        train_pa = list(set(self.pa_lines) - set(test_pa))
        train_pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in train_pa:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            train_pa_adj_matrix[row][col] = 1
        pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in self.pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1
        train_ac_adj_matrix = np.matmul(train_pa_adj_matrix.transpose(), pc_adj_matrix)
        row, col = np.nonzero(train_ac_adj_matrix)
        with open('../data/dblp/oriData/train_ac_pos_neg.txt', 'w') as t_aa_p_n_f:
            for i in xrange(len(row)):
                pos_a = row[i]
                pos_c = col[i]
                neg_c_list = list(np.where(train_ac_adj_matrix[pos_a, :] == 0)[0])
                neg_c_k_list = random.sample(neg_c_list, neg_num)
                neg_c = ' '.join(map(lambda x: str(x), neg_c_k_list))
                t_aa_p_n_f.write(str(pos_a) + '\t' + str(pos_c) + '\t' + neg_c + '\n')

    def get_good_author_4_test(self):
        # 选择训练集和测试集都出现的 author！！
        print('get good test pa...')
        test_a = map(lambda x:x.strip().split('\t')[1], self.test_pa)

        train_pa = list(set(self.pa_lines) - set(self.test_pa))
        train_a = map(lambda x:x.strip().split('\t')[1],train_pa)

        good_a = list(set(train_a) & set(test_a))
        self.good_author = good_a

        with open('../data/dblp_lp/good_author_4_test.txt', 'w') as t_pa_f:
            for a in good_a:
                t_pa_f.write(str(a)+'\n')


if __name__ == '__main__':
    dh = DataHelper('../data/dblp/oriData/')
    dh.load_data()

    dh.split_data_apa()

    dh.get_good_author_4_test()

    # # link prediction
    dh.get_test_aa_data()
    dh.get_train_aa_data()

    dh.get_data_4_our_symmetrical()
    dh.get_data_4_our_asymmetric()

    dh.get_data_4_baselines()
    dh.get_data_4_m2v('apa',10,100)
    dh.get_data_4_m2v('apcpa',10,100)
    dh.get_data_4_m2v('aptpa',10,100)
    dh.get_data_4_m2v('apa', 10, 50)
    dh.get_data_4_m2v('apcpa', 10, 50)
    dh.get_data_4_m2v('aptpa', 10, 50)

    # node recommendation
    # dh.get_test_ac_data(neg_num=1)
    # dh.get_train_ac_data(neg_num=1)



