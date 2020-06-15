# coding: utf-8
# author: lu yf
# create date: 2018/8/7
# re-write: 0808,02:00

import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir,baseline_data_dir,our_data_dir):
        self.dblp_data_fold = data_dir
        self.baseline_data_fold= baseline_data_dir
        self.our_data_fold = our_data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []

        self.pa_lines = []
        self.pc_lines = []
        self.pt_lines = []

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
            paper_name = 'p' + token[0]
            author_name = 'a' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(author_name):
                self.node2id[author_name] = len(self.node2id)
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
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))

        self.pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pa_adj_mtx[row][col] = 1
        self.apa_adj_mtx = np.matmul(self.pa_adj_mtx.transpose(),self.pa_adj_mtx)

        print ('#paper:{}, #author:{}, #conf:{}, #term:{}, #node: {}'.
               format(len(self.paper_list), len(self.author_list),
                      len(self.conf_list), len(self.term_list), len(self.node2id)))

    def split_data_three_parts(self):
        print('split with no_iso...')
        pa_adj_mtx = self.pa_adj_mtx
        target_num = 0.2 * len(self.pa_lines)
        self.test_pa_lines = set()
        while 1:
            cand_pa = random.choice(self.pa_lines)
            tokens = cand_pa.split('\t')
            cand_p = tokens[0]
            cand_a = tokens[1]
            if sum(pa_adj_mtx[:, int(cand_a)]) > 1:  # at least two papers
                self.test_pa_lines.add(cand_pa)
                pa_adj_mtx[int(cand_p)][int(cand_a)] = 0
            if len(self.test_pa_lines) >= target_num:
                break
        self.train_pa_lines = list(set(self.pa_lines) - set(self.test_pa_lines))
        self.train_paper = map(lambda x:x.strip().split('\t')[0],self.train_pa_lines)
        self.train_paper = list(set(self.train_paper))
        print('#train pa: {}, #test pa: {}, #train paper:{}'.
              format(len(self.train_pa_lines), len(self.test_pa_lines),len(self.train_paper)))

        self.process_data_4_lp()

        total_paper = len(self.paper_list)
        current_paper_list = self.train_paper
        for i in xrange(11):  # time steps == 10
            current_data_dict = self.process_data_4_dynamic(current_paper_list,
                                                            10 - i)  # start from t0 to t9, t10 is all nodes
            self.get_data_4_our_symmetrical(current_data_dict, 10 - i)
            self.get_data_4_dane(current_data_dict, 10-i)
            if i == 0 or i == 10:
                self.get_data_4_baselines(current_data_dict, 10 - i)
                self.get_data_4_m2v('apa', 10, 50, current_data_dict, 10 - i)
                self.get_data_4_m2v('apcpa', 10, 50, current_data_dict, 10 - i)
                self.get_data_4_m2v('aptpa', 10, 50, current_data_dict, 10 - i)

            delta_paper = random.sample(current_paper_list, int(0.001 * total_paper))
            current_paper_list = list(set(current_paper_list) - set(delta_paper))

            print('#time step: {}, #current paper: {}'.format(10-i, len(current_paper_list)))

    def process_data_4_lp(self):
        train_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        test_pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in self.pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            if line in self.train_pa_lines:
                train_pa_adj_mtx[row][col] = 1
            elif line in self.test_pa_lines:
                test_pa_adj_mtx[row][col] = 1
        self.get_aa_pos_neg_data('train',train_pa_adj_mtx)
        self.get_aa_pos_neg_data('test',test_pa_adj_mtx)

    def get_aa_pos_neg_data(self,data_type,pa_adj_mtx):
        """
        get positive and negative co-author data for link prediction
        :param data_type: train or test
        :param pa_adj_mtx:
        :return:
        """
        print('get {} aa data...'.format(data_type))
        apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pa_adj_mtx)
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

    def process_data_4_dynamic(self,current_paper_list,time_step):
        """
        process data for baselines and our
        get paper_author dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        paper_author = {}
        author_paper = {}
        paper_conf = {}
        conf_paper = {}
        paper_term = {}
        term_paper = {}

        print ('process data at time step: {}'.format(time_step))
        pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        with open(os.path.join('../data/dblpDynamic_lp/oriData', 'paper_author_'+str(time_step)+'.txt'),'w') as pa_file:
            for line in self.train_pa_lines:  # 在 train 上划分时间步!!!!
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pa_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pa_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    author_name = 'a' + token[1]
                    if not paper_author.has_key(paper_name):
                        paper_author[paper_name] = set()
                    paper_author[paper_name].add(author_name)
                    if not author_paper.has_key(author_name):
                        author_paper[author_name] = set()
                    author_paper[author_name].add(paper_name)

        pc_adj_mtx = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        with open(os.path.join('../data/dblpDynamic_lp/oriData', 'paper_conf_'+str(time_step)+'.txt'),'w') as pc_file:
            for line in self.pc_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pc_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pc_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    conf_name = 'c' + token[1]
                    if not paper_conf.has_key(paper_name):
                        paper_conf[paper_name] = set()
                    paper_conf[paper_name].add(conf_name)
                    if not conf_paper.has_key(conf_name):
                        conf_paper[conf_name] = set()
                    conf_paper[conf_name].add(paper_name)

        pt_adj_mtx = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        with open(os.path.join('../data/dblpDynamic_lp/oriData', 'paper_term_'+str(time_step)+'.txt'),'w') as pt_file:
            for line in self.pt_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pt_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pt_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    term_name = 't' + token[1]
                    if not paper_term.has_key(paper_name):
                        paper_term[paper_name] = set()
                    paper_term[paper_name].add(term_name)
                    if not term_paper.has_key(term_name):
                        term_paper[term_name] = set()
                    term_paper[term_name].add(paper_name)

        return {'pa':paper_author,'ap':author_paper,
                'pc':paper_conf,'cp':conf_paper,'pt':paper_term,'tp':term_paper,
                'pa_adj_mtx':pa_adj_mtx,'pc_adj_mtx':pc_adj_mtx,'pt_adj_mtx':pt_adj_mtx}

    def get_data_4_baselines(self,current_data_dict,time_step):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        paper_author = current_data_dict['pa']
        author_paper = current_data_dict['ap']
        paper_conf = current_data_dict['pc']
        conf_paper = current_data_dict['cp']
        paper_term = current_data_dict['pt']
        term_paper = current_data_dict['tp']
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_data_fold, 'dw.adjlist_lp_'+str(time_step)), 'w') as adj_file:
            for node_name, node_id in self.node2id.items():
                adj_line = str(node_id)
                next_node_list = []
                if node_name[0] == 'a':
                    if author_paper.has_key(node_name):
                        next_node_list = list(author_paper[node_name])
                elif node_name[0] == 'p':
                    if paper_term.has_key(node_name):
                        next_node_list += list(paper_term[node_name])
                    if paper_conf.has_key(node_name):
                        next_node_list += list(paper_conf[node_name])
                    if paper_author.has_key(node_name):
                        next_node_list += list(paper_author[node_name])
                elif node_name[0] == 'c':
                    if conf_paper.has_key(node_name):
                        next_node_list = list(conf_paper[node_name])
                elif node_name[0] == 't':
                    if term_paper.has_key(node_name):
                        next_node_list = list(term_paper[node_name])

                for n_n_name in next_node_list:
                    n_n_id = self.node2id[n_n_name]
                    adj_line += ' ' + str(n_n_id)
                adj_file.write(adj_line + '\n')

        # data for line
        print ('get data for line at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_data_fold, 'line.edgelist_lp_'+str(time_step)), 'w') as edge_file:
            for node_name, node_id in self.node2id.items():
                next_node_list = []
                if node_name[0] == 'a':
                    if author_paper.has_key(node_name):
                        next_node_list = list(author_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        continue
                elif node_name[0] == 'p':
                    if paper_term.has_key(node_name):
                        next_node_list += list(paper_term[node_name])
                    if paper_conf.has_key(node_name):
                        next_node_list += list(paper_conf[node_name])
                    if paper_author.has_key(node_name):
                        next_node_list += list(paper_author[node_name])
                elif node_name[0] == 'c':
                    if conf_paper.has_key(node_name):
                        next_node_list = list(conf_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        continue
                elif node_name[0] == 't':
                    if term_paper.has_key(node_name):
                        next_node_list = list(term_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        continue

                for n_n_name in next_node_list:
                    n_n_id = self.node2id[n_n_name]
                    edge_line = str(node_id) + ' ' + str(n_n_id) + ' ' + str(1)
                    edge_file.write(edge_line + '\n')

        # data for esim
        print('get data for esim at time step: {}'.format(time_step))
        with open(os.path.join(self.baseline_data_fold, 'esim.metapath_lp_'+str(time_step)), 'w') as metapath_file:
            metapath_file.write('apa 1' + '\n')
            metapath_file.write('apcpa 1'+'\n')
            metapath_file.write('aptpa 1')
            metapath_file.write('\n')
        with open(os.path.join(self.baseline_data_fold, 'esim.node_lp_'+str(time_step)), 'w') as node_file:
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
        with open(os.path.join(self.baseline_data_fold, 'esim.link_lp_'+str(time_step)), 'w') as net_file:
            for a, p_list in author_paper.items():
                for p in list(p_list):
                    net_file.write(a + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
                    net_file.write(p + ' ')
                    net_file.write(a + ' ')
                    net_file.write('\n')
            for p, c_list in paper_conf.items():
                for c in list(c_list):
                    net_file.write(p + ' ')
                    net_file.write(c + ' ')
                    net_file.write('\n')
                    net_file.write(c + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')
            for p, t_list in paper_term.items():
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
        author_paper = current_data_dict['ap']
        conf_paper = current_data_dict['cp']
        term_paper = current_data_dict['tp']
        dblp_adj_matrix = np.zeros([len(self.node2id), len(self.node2id)], dtype=float)
        for a, p_list in author_paper.items():
            for p in p_list:
                dblp_adj_matrix[self.node2id[a]][self.node2id[p]] = 1
                dblp_adj_matrix[self.node2id[p]][self.node2id[a]] = 1
        for c, p_list in conf_paper.items():
            for p in p_list:
                dblp_adj_matrix[self.node2id[c]][self.node2id[p]] = 1
                dblp_adj_matrix[self.node2id[p]][self.node2id[c]] = 1
        for t, p_list in term_paper.items():
            for p in p_list:
                dblp_adj_matrix[self.node2id[t]][self.node2id[p]] = 1
                dblp_adj_matrix[self.node2id[p]][self.node2id[t]] = 1
        dblp_csr_mtx = csr_matrix(dblp_adj_matrix)
        scipy.io.savemat(os.path.join('../baseline/dblpDynamic_lp/', 'dane_csr_lp_' + str(time_step) + '.mat'),
                         {'dblp_' + str(time_step): dblp_csr_mtx})

    def get_data_4_m2v(self, metapath, num_walks, walk_length,current_data_dict,time_step):
        """
        get data for metapath2vec
        over all authors!!
        :return:
        """
        paper_author = current_data_dict['pa']
        author_paper = current_data_dict['ap']
        paper_conf = current_data_dict['pc']
        conf_paper = current_data_dict['cp']
        paper_term = current_data_dict['pt']
        term_paper = current_data_dict['tp']
        # data for metapath2vec
        print('get data for metapath2vec at time step: {}'.format(time_step))
        print ('generating paths randomly via {}...'.format(metapath))
        file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + \
                    str(walk_length) + '_paths_lp_'+str(time_step)+'.txt'
        outfile = open(os.path.join(self.baseline_data_fold, file_name), 'w')
        for j in xrange(0, num_walks):  # wnum walks
            for author in self.author_list:
                outline = 'a' + str(author)
                author = 'a' + str(author)
                for i in xrange(walk_length):
                    if metapath == 'apa':
                        # select 'p'
                        if author_paper.has_key(author):
                            next_p_list = list(author_paper[author])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if paper_author.has_key(next_p_node):
                            next_a_list = list(paper_author[next_p_node])
                        else:
                            next_a_list = map(lambda x: 'a' + x, self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'apcpa':
                        # select 'p'
                        if author_paper.has_key(author):
                            next_p_list = list(author_paper[author])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'c'
                        if paper_conf.has_key(next_p_node):
                            next_c_list = list(paper_conf[next_p_node])
                        else:
                            next_c_list = map(lambda x: 'c' + x, self.conf_list)
                        next_c_node = random.choice(next_c_list)
                        outline += ' ' + next_c_node
                        # select 'p'
                        if conf_paper.has_key(next_c_node):
                            next_p_list = list(conf_paper[next_c_node])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if paper_author.has_key(next_p_node):
                            next_a_list = list(paper_author[next_p_node])
                        else:
                            next_a_list = map(lambda x: 'a' + x, self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node

                    elif metapath == 'aptpa':
                        # select 'p'
                        if author_paper.has_key(author):
                            next_p_list = list(author_paper[author])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 't'
                        if paper_term.has_key(next_p_node):
                            next_t_list = list(paper_term[next_p_node])
                        else:
                            next_t_list = map(lambda x: 't' + x, self.term_list)
                        next_t_node = random.choice(next_t_list)
                        outline += ' ' + next_t_node
                        # select 'p'
                        if term_paper.has_key(next_t_node):
                            next_p_list = list(term_paper[next_t_node])
                        else:
                            next_p_list = map(lambda x: 'p' + x, self.paper_list)
                        next_p_node = random.choice(next_p_list)
                        outline += ' ' + str(next_p_node)
                        # select 'a'
                        if paper_author.has_key(next_p_node):
                            next_a_list = list(paper_author[next_p_node])
                        else:
                            next_a_list = map(lambda x: 'a' + x, self.author_list)
                        next_a_node = random.choice(next_a_list)
                        outline += ' ' + str(next_a_node)
                        author = next_a_node
                outfile.write(outline + "\n")

        outfile.close()

    def get_data_4_our_symmetrical(self,current_data_dict,time_stem):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical) at time step: {}'.format(time_stem))
        pa_adj_mtx = current_data_dict['pa_adj_mtx']
        pc_adj_mtx = current_data_dict['pc_adj_mtx']
        pt_adj_mtx = current_data_dict['pt_adj_mtx']

        apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pa_adj_mtx)
        apa_csr_mtx = csr_matrix(apa_adj_mtx)
        self.save_mat(apa_csr_mtx, 'apa_csr_lp_'+str(time_stem))

        apc_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pc_adj_mtx)
        apcpa_adj_mtx = np.matmul(apc_adj_mtx,apc_adj_mtx.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_lp_'+str(time_stem))

        apt_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pt_adj_mtx)
        aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_lp_'+str(time_stem))

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
                    baseline_data_dir='../baseline/dblpDynamic_lp/',
                    our_data_dir='../data/dblpDynamic_lp')
    dh.load_data()

    dh.split_data_three_parts()