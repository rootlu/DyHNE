# coding: utf-8
# author: lu yf
# create date: 2018/8/7
import os
import numpy as np
import scipy.io
import random
import collections
from scipy.sparse import csr_matrix

# paper_num = 18181
# author_num = 22942
# conf_num = 22
# term_num = 8833
# year_num = 16
#
# # 处理aptpa矩阵 for our
# # for i in xrange(6):  # time steps == 5  2000--2005
# #     time_step = 2005 - i - 1
# #     print ('process data at time step: {}'.format(time_step))
# #     pa_adj_mtx = np.zeros([paper_num, author_num], dtype=float)
# #     with open(os.path.join('../data/aminerDynamic/oriData', 'paper_author_' + str(time_step) + '.txt'), 'r') as pa_file:
# #         for line in pa_file:
# #             token = line.strip('\n').split('\t')
# #             row = int(token[0])
# #             col = int(token[1])
# #             pa_adj_mtx[row][col] = 1
# #
# #     pt_adj_mtx = np.zeros([paper_num, term_num], dtype=float)
# #     with open(os.path.join('../data/aminerDynamic/oriData', 'paper_term_' + str(time_step) + '.txt'), 'r') as pt_file:
# #         for line in pt_file:
# #             token = line.strip('\n').split('\t')
# #             row = int(token[0])
# #             col = int(token[1])
# #             pt_adj_mtx[row][col] = 1
# #
# #     apt_adj_mtx = np.matmul(pa_adj_mtx.transpose(), pt_adj_mtx)
# #     aptpa_adj_mtx = np.matmul(apt_adj_mtx, apt_adj_mtx.transpose())
# #     aptpa_csr_mtx = csr_matrix(aptpa_adj_mtx)
# #     scipy.io.savemat(os.path.join('../data/aminerDynamic/', 'aptpa_csr_' + str(time_step)),
# #                      {'aptpa_csr_' + str(time_step): aptpa_csr_mtx})
#
# # # 处理初始网络 for metatpah2vec
# # paper_author = {}
# # author_paper = {}
# # paper_conf = {}
# # conf_paper = {}
# # paper_term = {}
# # term_paper = {}
# # with open(os.path.join('../data/aminerDynamic/oriData', 'paper_author_2000.txt'),'r') as pa_file:
# #     for line in pa_file:
# #         token = line.strip('\n').split('\t')
# #         paper_name = 'p' + token[0]
# #         author_name = 'a' + token[1]
# #         if not paper_author.has_key(paper_name):
# #             paper_author[paper_name] = set()
# #         paper_author[paper_name].add(author_name)
# #         if not author_paper.has_key(author_name):
# #             author_paper[author_name] = set()
# #         author_paper[author_name].add(paper_name)
# # with open(os.path.join('../data/aminerDynamic/oriData', 'paper_conf_2000.txt'),'r') as pc_file:
# #     for line in pc_file:
# #         token = line.strip('\n').split('\t')
# #         paper_name = 'p' + token[0]
# #         conf_name = 'c' + token[1]
# #         if not paper_conf.has_key(paper_name):
# #             paper_conf[paper_name] = set()
# #         paper_conf[paper_name].add(conf_name)
# #         if not conf_paper.has_key(conf_name):
# #             conf_paper[conf_name] = set()
# #         conf_paper[conf_name].add(paper_name)
# #
# # with open(os.path.join('../data/aminerDynamic/oriData', 'paper_term_2000.txt'), 'r') as pt_file:
# #     for line in pt_file:
# #         token = line.strip('\n').split('\t')
# #         row = int(token[0])
# #         col = int(token[1])
# #         paper_name = 'p' + token[0]
# #         term_name = 't' + token[1]
# #         if not paper_term.has_key(paper_name):
# #             paper_term[paper_name] = set()
# #         paper_term[paper_name].add(term_name)
# #         if not term_paper.has_key(term_name):
# #             term_paper[term_name] = set()
# #         term_paper[term_name].add(paper_name)
# #
# #
# # def get_data_4_m2v(metapath, num_walks, walk_length, time_step):
# #     """
# #     get data for metapath2vec
# #     over all authors!!
# #     :return:
# #     """
# #     # data for metapath2vec
# #     print('get data for metapath2vec at time step: {}'.format(time_step))
# #     print ('generating paths randomly via {}...'.format(metapath))
# #     file_name = 'm2v_' + metapath + '_' + 'w' + str(num_walks) + '_l' + \
# #                 str(walk_length) + '_paths_' + str(time_step) + '.txt'
# #     outfile = open(os.path.join('../baseline/aminerDynamic', file_name), 'w')
# #     for j in xrange(0, num_walks):  # wnum walks
# #         for author in xrange(author_num):
# #             outline = 'a' + str(author)
# #             author = 'a' + str(author)
# #             for i in xrange(walk_length):
# #                 if metapath == 'apa':
# #                     # select 'p'
# #                     if author_paper.has_key(author):
# #                         next_p_list = list(author_paper[author])
# #                         next_p_node = random.choice(next_p_list)
# #                     else:
# #                         next_p_node = 'p' + str(random.randrange(paper_num))
# #                     outline += ' ' + str(next_p_node)
# #                     # select 'a'
# #                     if paper_author.has_key(next_p_node):
# #                         next_a_list = list(paper_author[next_p_node])
# #                         next_a_node = random.choice(next_a_list)
# #                     else:
# #                         next_a_node = 'a' + str(random.randrange(author_num))
# #                     outline += ' ' + str(next_a_node)
# #                     author = next_a_node
# #
# #                 elif metapath == 'apcpa':
# #                     # select 'p'
# #                     if author_paper.has_key(author):
# #                         next_p_list = list(author_paper[author])
# #                         next_p_node = random.choice(next_p_list)
# #                     else:
# #                         next_p_node = 'p' + str(random.randrange(paper_num))
# #                     outline += ' ' + str(next_p_node)
# #                     # select 'c'
# #                     if paper_conf.has_key(next_p_node):
# #                         next_c_list = list(paper_conf[next_p_node])
# #                         next_c_node = random.choice(next_c_list)
# #                     else:
# #                         next_c_node = 'c' + str(random.randrange(conf_num))
# #                     outline += ' ' + next_c_node
# #                     # select 'p'
# #                     if conf_paper.has_key(next_c_node):
# #                         next_p_list = list(conf_paper[next_c_node])
# #                         next_p_node = random.choice(next_p_list)
# #                     else:
# #                         next_p_node = 'p' + str(random.randrange(paper_num))
# #                     outline += ' ' + str(next_p_node)
# #                     # select 'a'
# #                     if paper_author.has_key(next_p_node):
# #                         next_a_list = list(paper_author[next_p_node])
# #                         next_a_node = random.choice(next_a_list)
# #                     else:
# #                         next_a_node = 'a' + str(random.randrange(author_num))
# #                     outline += ' ' + str(next_a_node)
# #                     author = next_a_node
# #
# #                 elif metapath == 'aptpa':
# #                     # select 'p'
# #                     if author_paper.has_key(author):
# #                         next_p_list = list(author_paper[author])
# #                         next_p_node = random.choice(next_p_list)
# #                     else:
# #                         next_p_node = 'p' + str(random.randrange(paper_num))
# #                     outline += ' ' + str(next_p_node)
# #                     # select 't'
# #                     if paper_term.has_key(next_p_node):
# #                         next_t_list = list(paper_term[next_p_node])
# #                         next_t_node = random.choice(next_t_list)
# #                     else:
# #                         next_t_node = 't' + str(random.randrange(term_num))
# #                     outline += ' ' + next_t_node
# #                     # select 'p'
# #                     if term_paper.has_key(next_t_node):
# #                         next_p_list = list(term_paper[next_t_node])
# #                         next_p_node = random.choice(next_p_list)
# #                     else:
# #                         next_p_node = 'p' + str(random.randrange(paper_num))
# #                     outline += ' ' + str(next_p_node)
# #                     # select 'a'
# #                     if paper_author.has_key(next_p_node):
# #                         next_a_list = list(paper_author[next_p_node])
# #                         next_a_node = random.choice(next_a_list)
# #                     else:
# #                         next_a_node = 'a' + str(random.randrange(author_num))
# #                     outline += ' ' + str(next_a_node)
# #                     author = next_a_node
# #             outfile.write(outline + "\n")
# #
# #     outfile.close()
# #
# #
# # # get_data_4_m2v('apa', 10, 50, 2000)
# # # get_data_4_m2v('apcpa', 10, 50, 2000)
# # get_data_4_m2v('aptpa', 10, 50, 2000)
#
# # 处理初始网络 for dw, line
# paper_author = {}
# author_paper = {}
# paper_conf = {}
# conf_paper = {}
# paper_term = {}
# term_paper = {}
# paper_year = {}
# year_paper = {}
# with open(os.path.join('../data/aminerDynamic/oriData', 'paper_author_2004.txt'), 'r') as pa_file:
#     for line in pa_file:
#         token = line.strip('\n').split('\t')
#         paper_name = 'p' + token[0]
#         author_name = 'a' + token[1]
#         if not paper_author.has_key(paper_name):
#             paper_author[paper_name] = set()
#         paper_author[paper_name].add(author_name)
#         if not author_paper.has_key(author_name):
#             author_paper[author_name] = set()
#         author_paper[author_name].add(paper_name)
# with open(os.path.join('../data/aminerDynamic/oriData', 'paper_conf_2004.txt'), 'r') as pc_file:
#     for line in pc_file:
#         token = line.strip('\n').split('\t')
#         paper_name = 'p' + token[0]
#         conf_name = 'c' + token[1]
#         if not paper_conf.has_key(paper_name):
#             paper_conf[paper_name] = set()
#         paper_conf[paper_name].add(conf_name)
#         if not conf_paper.has_key(conf_name):
#             conf_paper[conf_name] = set()
#         conf_paper[conf_name].add(paper_name)
#
# with open(os.path.join('../data/aminerDynamic/oriData', 'paper_term_2004.txt'), 'r') as pt_file:
#     for line in pt_file:
#         token = line.strip('\n').split('\t')
#         row = int(token[0])
#         col = int(token[1])
#         paper_name = 'p' + token[0]
#         term_name = 't' + token[1]
#         if not paper_term.has_key(paper_name):
#             paper_term[paper_name] = set()
#         paper_term[paper_name].add(term_name)
#         if not term_paper.has_key(term_name):
#             term_paper[term_name] = set()
#         term_paper[term_name].add(paper_name)
#
# with open(os.path.join('../data/aminerDynamic/oriData', 'paper_year_2004.txt'), 'r') as py_file:
#     for line in py_file:
#         token = line.strip('\n').split('\t')
#         row = int(token[0])
#         col = int(token[1])
#         paper_name = 'p' + token[0]
#         year_name = 'y' + token[1]
#         if not paper_year.has_key(paper_name):
#             paper_year[paper_name] = set()
#         paper_year[paper_name].add(year_name)
#         if not year_paper.has_key(year_name):
#             year_paper[year_name] = set()
#         year_paper[year_name].add(paper_name)
#
# with open('../baseline/aminerDynamic/dw.node2id_2004', 'r') as d_f:
#     node2id = {}
#     for line in d_f:
#         tokens = line.strip().split('\t')
#         node2id[tokens[0]] = tokens[1]
#
# # dw
# with open(os.path.join('../baseline/aminerDynamic/', 'dw.adjlist_2004_iso'), 'w') as adj_file:
#     for node_name, node_id in node2id.items():
#         adj_line = str(node_id)
#         next_node_list = []
#         if node_name[0] == 'a':
#             if author_paper.has_key(node_name):
#                 next_node_list = list(author_paper[node_name])
#             else:
#                 adj_file.write(adj_line + '\n')
#                 continue
#         elif node_name[0] == 'p':
#             if paper_term.has_key(node_name):
#                 next_node_list += list(paper_term[node_name])
#             if paper_year.has_key(node_name):
#                 next_node_list += list(paper_year[node_name])
#             if paper_conf.has_key(node_name):
#                 next_node_list += list(paper_conf[node_name])
#             if paper_author.has_key(node_name):
#                 next_node_list += list(paper_author[node_name])
#         elif node_name[0] == 'c':
#             if conf_paper.has_key(node_name):
#                 next_node_list = list(conf_paper[node_name])
#             else:
#                 adj_file.write(adj_line + '\n')
#                 continue
#         elif node_name[0] == 't':
#             if term_paper.has_key(node_name):
#                 next_node_list = list(term_paper[node_name])
#             else:
#                 adj_file.write(adj_line + '\n')
#                 continue
#         elif node_name[0] == 'y':
#             if year_paper.has_key(node_name):
#                 next_node_list = list(year_paper[node_name])
#             else:
#                 adj_file.write(adj_line + '\n')
#                 continue
#
#         for n_n_name in next_node_list:
#             n_n_id = node2id[n_n_name]
#             adj_line += ' ' + str(n_n_id)
#         adj_file.write(adj_line + '\n')
#
# # data for line
# with open(os.path.join('../baseline/aminerDynamic/', 'line.edgelist_2004_iso'), 'w') as edge_file:
#     for node_name, node_id in node2id.items():
#         next_node_list = []
#         if node_name[0] == 'a':
#             if author_paper.has_key(node_name):
#                 next_node_list = list(author_paper[node_name])
#             else:
#                 edge_file.write(str(node_id) + '\n')
#                 continue
#                 # next_node_list = map(lambda x: 'p' + x, self.paper_list)
#         elif node_name[0] == 'p':
#             if paper_term.has_key(node_name):
#                 next_node_list += list(paper_term[node_name])
#             if paper_year.has_key(node_name):
#                 next_node_list += list(paper_year[node_name])
#             if paper_conf.has_key(node_name):
#                 next_node_list += list(paper_conf[node_name])
#             if paper_author.has_key(node_name):
#                 next_node_list += list(paper_author[node_name])
#         elif node_name[0] == 'c':
#             if conf_paper.has_key(node_name):
#                 next_node_list = list(conf_paper[node_name])
#             else:
#                 edge_file.write(str(node_id) + '\n')
#                 continue
#         elif node_name[0] == 't':
#             if term_paper.has_key(node_name):
#                 next_node_list = list(term_paper[node_name])
#             else:
#                 edge_file.write(str(node_id) + '\n')
#                 continue
#         elif node_name[0] == 'y':
#             if year_paper.has_key(node_name):
#                 next_node_list = list(year_paper[node_name])
#             else:
#                 edge_file.write(str(node_id) + '\n')
#                 continue
#
#         for n_n_name in next_node_list:
#             n_n_id = node2id[n_n_name]
#             edge_line = str(node_id) + ' ' + str(n_n_id) + ' ' + str(1)
#             edge_file.write(edge_line + '\n')

data_fold = '../data/yelpWWW/oriData/'
row = []
col = []
data = []
ur_adj_matrix = np.zeros([1286,33360])
with open(os.path.join(data_fold, 'user_review.txt')) as ur_file:
    for line in ur_file:
        token = line.strip().split('\t')
        row.append(int(token[0]))
        col.append(int(token[1]))
        data.append(1)
        ur_adj_matrix[int(token[0])][int(token[1])] = 1
    # ur_adj_matrix = csr_matrix((np.array(data), (row, col)),shape=(1286, 33360))

row = []
col = []
data = []
br_adj_matrix = np.zeros([2614,33360])
with open(os.path.join(data_fold, 'business_review.txt')) as br_file:
    for line in br_file:
        token = line.strip().split('\t')
        row.append(int(token[0]))
        col.append(int(token[1]))
        data.append(1)
        br_adj_matrix[int(token[0])][int(token[1])] = 1
    # br_adj_matrix = csr_matrix((np.array(data), (row, col)),shape=(2614, 33360))

# brurb = br_adj_matrix * ur_adj_matrix.transpose() *\
#         ur_adj_matrix * br_adj_matrix.transpose()

brurb = np.matmul(np.matmul(br_adj_matrix,ur_adj_matrix.transpose()),np.matmul(ur_adj_matrix,br_adj_matrix.transpose()))

train_num = 402574
test_num = 45451
row,col = np.where(brurb == 0)
neg_bb_list = zip(row,col)
train_neg_bb = random.sample(neg_bb_list,train_num)
test_neg_bb = random.sample(neg_bb_list,test_num)
print('save...')
with open(os.path.join('../data/yelpWWW_lp', 'neg_brurb_train.txt'), 'w') as neg_bb_train_file:
    for t_n_bb in train_neg_bb:
        neg_bb_train_file.write(str(t_n_bb[0]) + '\t' + str(t_n_bb[1]) + '\t' + '\n')

with open(os.path.join('../data/yelpWWW_lp', 'neg_brurb_test.txt'), 'w') as neg_bb_test_file:
    for t_n_bb in test_neg_bb:
        neg_bb_test_file.write(str(t_n_bb[0]) + '\t' + str(t_n_bb[1]) + '\t' + '\n')





