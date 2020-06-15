# coding: utf-8
# author: lu yf
# create date: 2018/8/8

import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix
from tqdm import tqdm
import networkx as nx


class DataHelper:
    def __init__(self,data_dir,baseline_data_dir,our_data_dir):
        self.dblp_data_fold = data_dir
        self.baseline_data_fold = baseline_data_dir
        self.our_data_fold = our_data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []

        self.pa_lines = []
        self.pc_lines = []
        self.pt_lines = []

        self.node2id = {}
        self.id2node = {}
        self.G = nx.Graph()
        self.train_G = nx.Graph()
        self.test_G = nx.Graph()

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
            self.G.add_edge(self.node2id[paper_name],self.node2id[author_name])
            self.G.add_node(self.node2id[paper_name],node_type='p')
            self.G.add_node(self.node2id[author_name],node_type='a')

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
            self.G.add_edge(self.node2id[paper_name],self.node2id[conf_name])
            self.G.add_node(self.node2id[paper_name],node_type='p')
            self.G.add_node(self.node2id[conf_name],node_type='c')

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
            self.G.add_edge(self.node2id[paper_name],self.node2id[term_name])
            self.G.add_node(self.node2id[paper_name],node_type='p')
            self.G.add_node(self.node2id[term_name],node_type='t')

        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))

        pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_mtx[row][col] = 1
        self.apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pa_adj_mtx)

        self.save_node2id_id2node()
        if self.G.number_of_edges() == len(self.pa_lines)+len(self.pc_lines)+len(self.pt_lines):
            print ('#paper:{}, #author:{}, #conf:{}, #term:{}, #node: {}, #edge: {}'.
                   format(len(self.paper_list), len(self.author_list),
                          len(self.conf_list), len(self.term_list),
                          len(self.node2id), (len(self.pa_lines)+len(self.pc_lines)+len(self.pt_lines))))
        else:
            print('build network filed!')
            return

    def save_node2id_id2node(self):
        self.id2node = {v: k for k, v in self.node2id.items()}
        with open(os.path.join(self.baseline_data_fold, 'node2id_lp'), 'w') as n2id_file:
            for n_name, n_id in self.node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        with open(os.path.join(self.baseline_data_fold, 'id2node_lp'), 'w') as id2n_file:
            for n_id, n_name in self.id2node.items():
                id2n_file.write(str(n_id) + '\t' + n_name + '\n')

    def split_network(self,split_ratio):
        if nx.is_connected(self.G):
            print('network has no iso.')
        else:
            print('network has iso.')
        remove_edge_num = int(split_ratio*self.G.number_of_edges())
        print('split network with ratio {}, numbs: {}...'.format(split_ratio,remove_edge_num))
        # while 1:
        #     remove_edge_list = random.sample(list(self.G.edges),remove_edge_num)
        #     self.train_G = self.G.copy()
        #     self.train_G.remove_edges_from(remove_edge_list)
        #     if nx.is_connected(self.train_G):
        #         self.test_G.add_edges_from(remove_edge_list)
        #         break
        remove_edge_list = random.sample(list(self.G.edges), remove_edge_num)
        self.train_G = self.G.copy()
        self.train_G.remove_edges_from(remove_edge_list)
        self.test_G.add_edges_from(remove_edge_list)
        print('#node for train: {}, #edge for train:{}'.
              format(self.train_G.number_of_nodes(),self.train_G.number_of_edges()))
        print('#node for test: {}, #edge for test:{}'.
              format(self.test_G.number_of_nodes(), self.test_G.number_of_edges()))

    def get_data_4_homo_baselines(self):
        train_edges = list(self.train_G.edges)
        print ('get data for deepwalk or node2vec...')
        with open(os.path.join(self.baseline_data_fold, 'dw.edgelist_lp'), 'w') as edge_file:
            for e in train_edges:
                edge_file.write(str(e[0])+' '+str(e[1]) + '\n')
                edge_file.write(str(e[1])+' '+str(e[0]) + '\n')
        print ('get data for line...')
        with open(os.path.join(self.baseline_data_fold, 'line.edgelist_lp'), 'w') as edge_file:
            for e in train_edges:
                edge_file.write(str(e[0])+' '+str(e[1])+' '+str(1) + '\n')
                edge_file.write(str(e[1])+' '+str(e[0])+' '+str(1) + '\n')

    def get_hete_data(self):
        print('get hete data...')
        author_paper = {}
        paper_author = {}
        paper_conf = {}
        conf_paper = {}
        paper_term = {}
        term_paper = {}
        pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        pc_adj_mtx = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        pt_adj_mtx = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        train_author = []
        test_author = []

        print('train data...')
        train_edges = list(self.train_G.edges)
        for e in train_edges:
            node_1_name = self.id2node[e[0]]
            node_2_name = self.id2node[e[1]]
            relation_type = node_1_name[0]+node_2_name[0]
            # relation_type = self.train_G.edges[e]['rel']

            if relation_type == 'pa':
                if not paper_author.has_key(node_1_name):
                    paper_author[node_1_name] = []
                paper_author[node_1_name].append(node_2_name)
                pa_adj_mtx[int(node_1_name[1:])][int(node_2_name[1:])] = 1
            elif relation_type == 'ap':
                if not author_paper.has_key(node_1_name):
                    author_paper[node_1_name] = []
                author_paper[node_1_name].append(node_2_name)

            elif relation_type == 'pc':
                if not paper_conf.has_key(node_1_name):
                    paper_conf[node_1_name] = node_2_name
                pc_adj_mtx[int(node_1_name[1:])][int(node_2_name[1:])] = 1
            elif relation_type == 'cp':
                if not conf_paper.has_key(node_1_name):
                    conf_paper[node_1_name] = []
                conf_paper[node_1_name].append(node_2_name)

            elif relation_type == 'pt':
                if not paper_term.has_key(node_1_name):
                    paper_term[node_1_name] = []
                paper_term[node_1_name].append(node_2_name)
                pt_adj_mtx[int(node_1_name[1:])][int(node_2_name[1:])] = 1
            elif relation_type == 'tp':
                if not term_paper.has_key(node_1_name):
                    term_paper[node_1_name] = []
                term_paper[node_1_name].append(node_2_name)
            else:
                print('unknown relation: {}'.format(relation_type))

            if node_1_name[0] == 'a':
                train_author.append(int(node_1_name[1:]))
            elif node_2_name == 'a':
                train_author.append(int(node_2_name[1:]))
        train_author = list(set(train_author))
        print('#non-iso. author in train: {}'.format(len(train_author)))
        author_conf = {}
        conf_author = {}
        for a, p_list in author_paper.items():
            for p in p_list:
                if paper_conf.has_key(p):
                    c = paper_conf[p]
                    if not author_conf.has_key(a):
                        author_conf[a] = []
                    author_conf[a].append(c)
                    if not conf_author.has_key(c):
                        conf_author[c] = []
                    conf_author[c].append(a)

        author_term = {}
        term_author = {}
        for a, p_list in author_paper.items():
            for p in p_list:
                if paper_term.has_key(p):
                    t_list = paper_term[p]
                    for t in t_list:
                        if not author_term.has_key(a):
                            author_term[a] = []
                        author_term[a].append(t)
                        if not term_author.has_key(t):
                            term_author[t] = []
                        term_author[t].append(a)

        print('test data...')
        test_edges = list(self.test_G.edges)
        test_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for e in test_edges:
            node_1_name = self.id2node[e[0]]
            node_2_name = self.id2node[e[1]]
            relation_type = node_1_name[0]+node_2_name[0]
            if relation_type == 'pa':
                test_pa_adj_mtx[int(node_1_name[1:])][int(node_2_name[1:])] = 1

            if node_1_name[0] == 'a':
                test_author.append(int(node_1_name[1:]))
            elif node_2_name == 'a':
                test_author.append(int(node_2_name[1:]))
        test_author = list(set(test_author))
        print('#non-iso. author in test: {}'.format(len(test_author)))

        good_author = list(set(train_author) & set(test_author))
        print('good_author: {}'.format(len(good_author)))

        return {'pa': paper_author, 'ap': author_paper,
                'pc': paper_conf, 'cp': conf_paper,
                'pt': paper_term, 'tp': term_paper,
                'ac': author_conf, 'ca': conf_author,
                'at': author_term, 'ta': term_author,
                'pa_adj_mtx':pa_adj_mtx,'pc_adj_mtx':pc_adj_mtx,'pt_adj_mtx':pt_adj_mtx,
                'test_pa_adj_mtx':test_pa_adj_mtx,'good_author':good_author}

    def get_data_4_hete_baselines(self,hete_data):
        train_edges = list(self.train_G.edges)
        print('get data for esim...')
        with open(os.path.join(self.baseline_data_fold, 'esim.metapath_lp'), 'w') as metapath_file:
            metapath_file.write('apa 1' + '\n')
            metapath_file.write('apcpa 1' + '\n')
            metapath_file.write('aptpa 1' + '\n')
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
            for e in train_edges:
                node_1_name = self.id2node[e[0]]
                node_2_name = self.id2node[e[1]]
                net_file.write(node_1_name + ' ' + node_2_name + '\n')
                net_file.write(node_2_name + ' ' + node_1_name + '\n')

        print('get data for metapath2vec...')
        print ('apa...')
        outfile = open(os.path.join(self.baseline_data_fold, 'm2v_apa_w10_l50_paths_lp.txt'), 'w')
        for author in hete_data['ap']:
            for j in xrange(0, 10):
                outline = author
                for i in xrange(0, 50):
                    if hete_data['ap'].has_key(author):
                        next_p_list = list(hete_data['ap'][author])
                        next_p_node = random.choice(next_p_list)
                    else:
                        next_p_node = 'p'+random.choice(self.paper_list)
                    outline += ' ' + next_p_node
                    if hete_data['pa'].has_key(next_p_node):
                        next_a_list = list(hete_data['pa'][next_p_node])
                        next_a_node = random.choice(next_a_list)
                    else:
                        next_a_node = 'a'+random.choice(self.author_list)
                    outline += ' ' + next_a_node
                    author = next_a_node
                outfile.write(outline + "\n")
        outfile.close()

        print ('apcpa...')
        outfile = open(os.path.join(self.baseline_data_fold, 'm2v_apcpa_w10_l50_paths_lp.txt'), 'w')
        for author in hete_data['ap']:
            for j in xrange(0, 10):
                outline = author
                for i in xrange(0, 50):
                    if hete_data['ap'].has_key(author):
                        next_p_list = list(hete_data['ap'][author])
                        next_p_node = random.choice(next_p_list)
                    else:
                        continue
                    outline += ' ' + next_p_node
                    if hete_data['pc'].has_key(next_p_node):
                        next_c_list = list(hete_data['pc'][next_p_node])
                        next_c_node = random.choice(next_c_list)
                    else:
                        continue
                    outline += ' ' + next_c_node
                    if hete_data['cp'].has_key(next_c_node):
                        next_p_list = list(hete_data['cp'][next_c_node])
                        next_p_node = random.choice(next_p_list)
                    else:
                        continue
                    outline += ' ' + next_p_node
                    if hete_data['pa'].has_key(next_c_node):
                        next_a_list = list(hete_data['pa'][next_p_node])
                        next_a_node = random.choice(next_a_list)
                    else:
                        continue
                    outline += ' ' + next_a_node
                    author = next_a_node
                outfile.write(outline + "\n")
        outfile.close()

        print ('aptpa...')
        outfile = open(os.path.join(self.baseline_data_fold, 'm2v_aptpa_w10_l50_paths_lp.txt'), 'w')
        for author in hete_data['ap']:
            for j in xrange(0, 10):
                outline = author
                for i in xrange(0, 50):
                    if hete_data['ap'].has_key(author):
                        next_p_list = list(hete_data['ap'][author])
                        next_p_node = random.choice(next_p_list)
                    else:
                        continue
                    outline += ' ' + next_p_node
                    if hete_data['pt'].has_key(next_p_node):
                        next_t_list = list(hete_data['pt'][next_p_node])
                        next_t_node = random.choice(next_t_list)
                    else:
                        continue
                    outline += ' ' + next_t_node
                    if hete_data['tp'].has_key(next_t_node):
                        next_p_list = list(hete_data['tp'][next_t_node])
                        next_p_node = random.choice(next_p_list)
                    else:
                        continue
                    outline += ' ' + next_p_node
                    if hete_data['pa'].has_key(next_p_node):
                        next_a_list = list(hete_data['pa'][next_p_node])
                        next_a_node = random.choice(next_a_list)
                    else:
                        continue
                    outline += ' ' + next_a_node
                    author = next_a_node
                outfile.write(outline + "\n")
        outfile.close()

    def get_data_4_our(self,hete_data):
        print('get data for our (symmetrical)...')
        pa_adj_mtx = hete_data['pa_adj_mtx']
        pc_adj_mtx = hete_data['pc_adj_mtx']
        pt_adj_mtx = hete_data['pt_adj_mtx']

        apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pa_adj_mtx)
        apa_csr_mtx = csr_matrix(apa_adj_mtx)
        self.save_mat(apa_csr_mtx, 'apa_csr_lp')

        apc_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pc_adj_mtx)
        apcpa_adj_mtx = np.matmul(apc_adj_mtx, apc_adj_mtx.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_lp')

        apt_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pt_adj_mtx)
        aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_lp')

    def get_aa_pos_neg_data(self,data_type,hete_data,neg_aa):
        """
        get positive and negative co-author data for link prediction
        :return:
        """
        print('get {} aa data...'.format(data_type))
        if data_type == 'train':
            pa_adj_mtx = hete_data['pa_adj_mtx']
        elif data_type == 'test':
            pa_adj_mtx = hete_data['test_pa_adj_mtx']
        apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pa_adj_mtx)
        row, col = np.nonzero(np.triu(apa_adj_mtx))
        pos_num = 0
        with open(os.path.join(self.our_data_fold,data_type+'_aa_pos.txt'), 'w') as aa_pos_f:
            for i in xrange(len(row)):
                # if row[i] in hete_data['good_author'] and col[i] in hete_data['good_author']:
                pos_a_1 = row[i]
                pos_a_2 = col[i]
                aa_pos_f.write(str(pos_a_1)+'\t'+str(pos_a_2)+'\n')
                pos_num += 1
        print('#{} aa: {}'.format(data_type, pos_num))

        # negative apa
        neg_aa_list = random.sample(neg_aa,pos_num)

        with open(os.path.join(self.our_data_fold,data_type+'_aa_neg.txt'), 'w') as aa_neg_f:
            for n_aa in neg_aa_list:
                aa_neg_f.write(str(n_aa[0]) + '\t' + str(n_aa[1]) + '\n')

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
                    baseline_data_dir='../baseline/dblp_lp_nx/',
                    our_data_dir='../data/dblp_lp_nx')
    dh.load_data()
    # dh.build_network()
    dh.split_network(split_ratio=0.2)

    dh.get_data_4_homo_baselines()
    hete_data_dict = dh.get_hete_data()
    dh.get_data_4_hete_baselines(hete_data_dict)
    dh.get_data_4_our(hete_data_dict)
    neg_aa = np.where(dh.apa_adj_mtx == 0)
    cand_neg_aa_list = zip(neg_aa[0], neg_aa[1])
    dh.get_aa_pos_neg_data('train', hete_data_dict, cand_neg_aa_list)
    dh.get_aa_pos_neg_data('test', hete_data_dict, cand_neg_aa_list)
