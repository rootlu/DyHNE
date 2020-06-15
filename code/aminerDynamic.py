# coding: utf-8
# author: lu yf
# create date: 2018/8/3

from __future__ import division
import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix


class DataHelper:
    def __init__(self,data_dir):
        self.aminer_data_fold = data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []
        self.year_list = []

        self.pa_lines = []
        self.pc_lines = []
        self.pt_lines = []
        self.py_lines = []

        self.node2id = {}

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print ('loading data...')
        with open(os.path.join(self.aminer_data_fold, 'paper_author.txt')) as pa_file:
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
        with open(os.path.join(self.aminer_data_fold, 'paper_conf.txt')) as pc_file:
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
        with open(os.path.join(self.aminer_data_fold, 'paper_term.txt')) as pt_file:
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
        with open(os.path.join(self.aminer_data_fold, 'paper_year.txt')) as py_file:
            self.py_lines = py_file.readlines()
        for line in self.py_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.year_list.append(token[1])
            paper_name = 'p' + token[0]
            year_name = 'y' + token[1]
            if not self.node2id.has_key(paper_name):
                self.node2id[paper_name] = len(self.node2id)
            if not self.node2id.has_key(year_name):
                self.node2id[year_name] = len(self.node2id)
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))
        self.year_list = list(set(self.year_list))
        print ('#paper:{}, #author:{}, #conf:{}, #term: {}, #year:{}, #node:{}'.
               format(len(self.paper_list), len(self.author_list),
                      len(self.conf_list), len(self.term_list),
                      len(self.year_list), len(self.node2id)))

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
            # time step == 0  1990---2004
            static_paper_list,dynamic_paper_list = self.get_static_dynamic_paper_list(2004)
            static_data_dict = self.process_data(static_paper_list,2004)
            self.get_data_4_baselines(static_data_dict, 2004)
            self.get_data_4_our_symmetrical(static_data_dict,2004)
            self.get_data_4_m2v('apa',10,50,static_data_dict,2004)
            self.get_data_4_m2v('apcpa',10,50,static_data_dict,2004)
            self.get_data_4_m2v('aptpa',10,50,static_data_dict,2004)

            current_paper_list = static_paper_list
            delta_paper_num = len(dynamic_paper_list)/10

            for i in xrange(10):  # time steps == 10  2005year
                current_year = '2005_'+str(i+1)
                delta_paper = dynamic_paper_list[:int((i+1)*delta_paper_num)]
                current_paper_list = current_paper_list + delta_paper

                current_data_dict = self.process_data(current_paper_list,current_year)  # start from t0 to t9, t10 is all nodes

                self.get_data_4_our_symmetrical(current_data_dict,current_year)
                self.get_data_4_baselines(current_data_dict,current_year)

                print('#time setp: {}, #current paper: {}'.format(current_year, len(current_paper_list)))

    def get_static_dynamic_paper_list(self,year):
        static_paper_list = []
        dynamic_paper_list = []
        for line in self.py_lines:
            token = line.strip('\n').split('\t')
            if int(token[1]) <= year-1990:  # before year and year
                static_paper_list.append(token[0])
            else:
                dynamic_paper_list.append(token[0])
        return static_paper_list,dynamic_paper_list

    def process_data(self,current_paper_list,time_step):
        """
        process data for baselines and our
        get paper_author dict. and adj. matrix
        node2id includes all nodes!!!
        :return:
        """
        node2id = self.node2id
        paper_author = {}
        author_paper = {}
        paper_conf = {}
        conf_paper = {}
        paper_term = {}
        term_paper = {}
        paper_year = {}
        year_paper = {}

        print ('process data at time step: {}'.format(time_step))
        pa_adj_mtx = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        with open(os.path.join('../data/aminerDynamic/oriData', 'paper_author_'+str(time_step)+'.txt'),'w') as pa_file:
            for line in self.pa_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pa_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pa_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    author_name = 'a' + token[1]
                    # if not node2id.has_key(paper_name):
                    #     node2id[paper_name] = len(node2id)
                    # if not node2id.has_key(author_name):
                    #     node2id[author_name] = len(node2id)
                    if not paper_author.has_key(paper_name):
                        paper_author[paper_name] = set()
                    paper_author[paper_name].add(author_name)
                    if not author_paper.has_key(author_name):
                        author_paper[author_name] = set()
                    author_paper[author_name].add(paper_name)

        pc_adj_mtx = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        with open(os.path.join('../data/aminerDynamic/oriData', 'paper_conf_'+str(time_step)+'.txt'),'w') as pc_file:
            for line in self.pc_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pc_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pc_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    conf_name = 'c' + token[1]
                    # if not node2id.has_key(paper_name):
                    #     node2id[paper_name] = len(node2id)
                    # if not node2id.has_key(conf_name):
                    #     node2id[conf_name] = len(node2id)
                    if not paper_conf.has_key(paper_name):
                        paper_conf[paper_name] = set()
                    paper_conf[paper_name].add(conf_name)
                    if not conf_paper.has_key(conf_name):
                        conf_paper[conf_name] = set()
                    conf_paper[conf_name].add(paper_name)
        pt_adj_mtx = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        with open(os.path.join('../data/aminerDynamic/oriData', 'paper_term_' + str(time_step) + '.txt'), 'w') as pt_file:
            for line in self.pt_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    pt_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    pt_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    term_name = 't' + token[1]
                    # if not node2id.has_key(paper_name):
                    #     node2id[paper_name] = len(node2id)
                    # if not node2id.has_key(term_name):
                    #     node2id[term_name] = len(node2id)
                    if not paper_term.has_key(paper_name):
                        paper_term[paper_name] = set()
                    paper_term[paper_name].add(term_name)
                    if not term_paper.has_key(term_name):
                        term_paper[term_name] = set()
                    term_paper[term_name].add(paper_name)

        py_adj_mtx = np.zeros([len(self.paper_list), len(self.year_list)], dtype=float)
        with open(os.path.join('../data/aminerDynamic/oriData', 'paper_year_'+str(time_step)+'.txt'),'w') as py_file:
            for line in self.py_lines:
                token = line.strip('\n').split('\t')
                if token[0] in current_paper_list:
                    py_file.write(line)
                    row = int(token[0])
                    col = int(token[1])
                    py_adj_mtx[row][col] = 1
                    paper_name = 'p' + token[0]
                    year_name = 't' + token[1]
                    # if not node2id.has_key(paper_name):
                    #     node2id[paper_name] = len(node2id)
                    # if not node2id.has_key(year_name):
                    #     node2id[year_name] = len(node2id)
                    if not paper_year.has_key(paper_name):
                        paper_year[paper_name] = set()
                    paper_year[paper_name].add(year_name)
                    if not year_paper.has_key(year_name):
                        year_paper[year_name] = set()
                    year_paper[year_name].add(paper_name)

        return {'node2id':node2id,
                'pa':paper_author,'ap':author_paper,
                'pc':paper_conf,'cp':conf_paper,
                'pt':paper_term,'tp':term_paper,
                'py':paper_year,'yp':year_paper,
                'pa_adj_mtx':pa_adj_mtx,'pc_adj_mtx':pc_adj_mtx,
                'pt_adj_mtx':pt_adj_mtx,'py_adj_mtx':py_adj_mtx}

    def get_data_4_baselines(self,current_data_dict,time_step):
        """
        get data for baselines.
        include nodes not in train set
        :return:
        """
        node2id = self.node2id
        paper_author = current_data_dict['pa']
        author_paper = current_data_dict['ap']
        paper_conf = current_data_dict['pc']
        conf_paper = current_data_dict['cp']
        paper_term = current_data_dict['pt']
        term_paper = current_data_dict['tp']
        paper_year = current_data_dict['py']
        year_paper = current_data_dict['yp']
        # data for deepwalk, node2vec
        print ('get data for deepwalk or node2vec at time step: {}'.format(time_step))
        # with open(os.path.join('../baseline/aminerDynamic/', 'dw.edgelist_'+str(time_step)), 'w') as edge_file:
        #     for a, p_list in author_paper.items():
        #         for p in p_list:
        #             edge_file.write(str(self.node2id[a])+' '+str(self.node2id[p])+'\n')
        #             edge_file.write(str(self.node2id[p])+' '+str(self.node2id[a])+'\n')
        #     for c, p_list in conf_paper.items():
        #         for p in p_list:
        #             edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[p]) + '\n')
        #             edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[c]) + '\n')
        #     for t, p_list in term_paper.items():
        #         for p in p_list:
        #             edge_file.write(str(self.node2id[t]) + ' ' + str(self.node2id[p]) + '\n')
        #             edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[t]) + '\n')
        #     for y, p_list in year_paper.items():
        #         for p in p_list:
        #             edge_file.write(str(self.node2id[y]) + ' ' + str(self.node2id[p]) + '\n')
        #             edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[y]) + '\n')
        with open(os.path.join('../baseline/aminerDynamic/', 'dw.adjist_'+str(time_step)), 'w') as adj_file:
            for node_name, node_id in node2id.items():
                adj_line = str(node_id)
                next_node_list = []
                if node_name[0] == 'a':
                    if author_paper.has_key(node_name):
                        next_node_list = list(author_paper[node_name])
                    else:
                        adj_file.write(adj_line + '\n')
                        continue
                        # next_node_list = map(lambda x: 'p' + x, self.paper_list)
                elif node_name[0] == 'p':
                    if paper_term.has_key(node_name):
                        next_node_list += list(paper_term[node_name])
                    if paper_year.has_key(node_name):
                        next_node_list += list(paper_year[node_name])
                    if paper_conf.has_key(node_name):
                        next_node_list += list(paper_conf[node_name])
                    if paper_author.has_key(node_name):
                        next_node_list += list(paper_author[node_name])
                elif node_name[0] == 'c':
                    if conf_paper.has_key(node_name):
                        next_node_list = list(conf_paper[node_name])
                    else:
                        adj_file.write(adj_line + '\n')
                        continue
                elif node_name[0] == 't':
                    if term_paper.has_key(node_name):
                        next_node_list = list(term_paper[node_name])
                    else:
                        adj_file.write(adj_line + '\n')
                        continue
                elif node_name[0] == 'y':
                    if year_paper.has_key(node_name):
                        next_node_list = list(year_paper[node_name])
                    else:
                        adj_file.write(adj_line + '\n')
                        continue

                for n_n_name in next_node_list:
                    n_n_id = node2id[n_n_name]
                    adj_line += ' ' + str(n_n_id)
                adj_file.write(adj_line + '\n')

        with open(os.path.join('../baseline/aminerDynamic/', 'dw.node2id_'+str(time_step)), 'w') as n2id_file:
            for n_name, n_id in node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for line
        print ('get data for line at time setp: {}'.format(time_step))
        with open(os.path.join('../baseline/aminerDynamic/', 'line.edgelist_'+str(time_step)), 'w') as edge_file:
            # for a, p_list in author_paper.items():
            #     for p in p_list:
            #         edge_file.write(str(self.node2id[a]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
            #         edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[a]) + ' ' + str(1) + '\n')
            # for c, p_list in conf_paper.items():
            #     for p in p_list:
            #         edge_file.write(str(self.node2id[c]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
            #         edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[c]) + ' ' + str(1) + '\n')
            # for t, p_list in term_paper.items():
            #     for p in p_list:
            #         edge_file.write(str(self.node2id[t]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
            #         edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[t]) + ' ' + str(1) + '\n')
            # for y, p_list in year_paper.items():
            #     for p in p_list:
            #         edge_file.write(str(self.node2id[y]) + ' ' + str(self.node2id[p]) + ' ' + str(1) + '\n')
            #         edge_file.write(str(self.node2id[p]) + ' ' + str(self.node2id[y]) + ' ' + str(1) + '\n')
            for node_name, node_id in node2id.items():
                next_node_list = []
                if node_name[0] == 'a':
                    if author_paper.has_key(node_name):
                        next_node_list = list(author_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        continue
                        # next_node_list = map(lambda x: 'p' + x, self.paper_list)
                elif node_name[0] == 'p':
                    if paper_term.has_key(node_name):
                        next_node_list += list(paper_term[node_name])
                    if paper_year.has_key(node_name):
                        next_node_list += list(paper_year[node_name])
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
                elif node_name[0] == 'y':
                    if year_paper.has_key(node_name):
                        next_node_list = list(year_paper[node_name])
                    else:
                        edge_file.write(str(node_id) + '\n')
                        continue

                for n_n_name in next_node_list:
                    n_n_id = node2id[n_n_name]
                    edge_line = str(node_id) + ' ' + str(n_n_id) + ' ' + str(1)
                    edge_file.write(edge_line + '\n')

        with open(os.path.join('../baseline/aminerDynamic/', 'line.node2id_'+str(time_step)), 'w') as n2id_file:
            for n_name, n_id in node2id.items():
                n2id_file.write(n_name + '\t' + str(n_id) + '\n')

        # data for esim
        print('get data for esim at time step: {}'.format(time_step))
        with open(os.path.join('../baseline/aminerDynamic/', 'esim.metapath_'+str(time_step)), 'w') as metapath_file:
            metapath_file.write('apa 0.1' + '\n')
            metapath_file.write('apcpa 0.7'+'\n')
            metapath_file.write('aptpa 0.2')
            metapath_file.write('\n')
        with open(os.path.join('../baseline/aminerDynamic/', 'esim.node_'+str(time_step)), 'w') as node_file:
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
            for y in self.year_list:
                node_file.write('y' + y + ' y')
                node_file.write('\n')
        with open(os.path.join('../baseline/aminerDynamic/', 'esim.link_'+str(time_step)), 'w') as net_file:
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
            for p, y_list in paper_year.items():
                for y in list(y_list):
                    net_file.write(p + ' ')
                    net_file.write(y + ' ')
                    net_file.write('\n')
                    net_file.write(y + ' ')
                    net_file.write(p + ' ')
                    net_file.write('\n')

        # data for DANE
        print('get data for dane at time step: {}...'.format(time_step))
        aminer_adj_matrix = np.zeros([len(node2id), len(node2id)], dtype=float)
        with open('../baseline/aminerDynamic/line.edgelist_' + str(time_step), 'r') as edge_file:
            for edge in edge_file:
                tokens = edge.strip().split(' ')
                aminer_adj_matrix[int(tokens[0])][int(tokens[1])] = 1
        aminer_csr_mtx = csr_matrix(aminer_adj_matrix)
        scipy.io.savemat(os.path.join('../baseline/aminerDynamic/', 'dane_csr_' + str(time_step) + '.mat'),
                             {'aminer_csr_' + str(time_step): aminer_csr_mtx})

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
        paper_year = current_data_dict['py']
        year_paper = current_data_dict['yp']
        # data for metapath2vec
        print('get data for metapath2vec at time step: {}'.format(time_step))
        print ('generating paths randomly via {}...'.format(metapath))
        file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + \
                    str(walk_length) + '_paths_'+str(time_step)+'.txt'
        outfile = open(os.path.join('../baseline/aminerDynamic', file_name), 'w')
        for j in xrange(0, num_walks):  # wnum walks
            for author in self.author_list:
                outline = 'a' + str(author)
                author = 'a' + str(author)
                for i in xrange(walk_length):
                    if metapath == 'apa':
                        # select 'p'
                        if author_paper.has_key(author):
                            next_p_list = list(author_paper[author])
                        else:  # TODO: 可以优化！！！
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

    def get_data_4_our_symmetrical(self,current_data_dict,time_step):
        """
        get data for our model
        deal with sym. metapaths
        :return:
        """
        print('get data for our (symmetrical) at time step: {}'.format(time_step))
        pa_adj_mtx = current_data_dict['pa_adj_mtx']
        pc_adj_mtx = current_data_dict['pc_adj_mtx']
        pt_adj_mtx = current_data_dict['pt_adj_mtx']
        py_adj_mtx = current_data_dict['py_adj_mtx']

        apa_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pa_adj_mtx)
        apa_csr_mtx = csr_matrix(apa_adj_mtx)
        self.save_mat(apa_csr_mtx, 'apa_csr_'+str(time_step))

        apc_adj_mtx = np.matmul(pa_adj_mtx.transpose(),pc_adj_mtx)
        apcpa_adj_mtx = np.matmul(apc_adj_mtx,apc_adj_mtx.transpose())
        apcpa_csr_mtx = csr_matrix(apcpa_adj_mtx)
        self.save_mat(apcpa_csr_mtx, 'apcpa_csr_'+str(time_step))

        apt_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pt_adj_mtx)
        aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
        aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
        self.save_mat(aptpa_csr_mtx, 'aptpa_csr_' + str(time_step))

        apy_adj_mtx = np.matmul(pa_adj_mtx.transpose(),py_adj_mtx)
        apypa_adj_mtx = np.matmul(apy_adj_mtx, apy_adj_mtx.transpose())
        apypa_csr_mtx = csr_matrix(apypa_adj_mtx)
        self.save_mat(apypa_csr_mtx, 'apypa_csr_'+str(time_step))

    def save_mat(self, matrix, relation_name):
        """
        save data to mat
        :param matrix:
        :param relation_name:
        :return:
        """
        scipy.io.savemat(os.path.join('../data/aminerDynamic/', relation_name),
                         {relation_name: matrix})


if __name__ == '__main__':
    dh = DataHelper('../data/aminer/oriData/')
    dh.load_data()

    dh.split_data(load_from_file=False)