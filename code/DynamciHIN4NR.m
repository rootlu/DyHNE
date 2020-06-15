clc;clear;
addpath(genpath(pwd));

load ./data/dblp_nr/pc_csr_nr.mat;
load ./data/dblp_nr/apc_csr_nr.mat;
load ./data/dblp_nr/apa_csr_nr.mat;
load ./data/dblp_nr/apcpa_csr_nr.mat;
load ./data/dblp_nr/aptpa_csr_nr.mat;

W_pc = pc_csr_nr;
W_apc = apc_csr_nr;
W_apa = apa_csr_nr;
W_apcpa = apcpa_csr_nr;
W_aptpa = aptpa_csr_nr;


k = 100;
gamma = 1;

% obtain diagonal and laplacian matrix

W_unify = W_apcpa;
dunify = sum(W_unify,2);
D_unify = diag(dunify);
L_unify = D_unify- W_unify;  
W_unify = NormalizeAdj(W_unify,0,2);

t1=clock;
M_unify = (eye(size(W_unify,1)) - W_unify)' * (eye(size(W_unify,1)) - W_unify);
t2=clock;
fprintf('Time for M_unify: %f s \n', etime(t2,t1));

[a_unify_embedding, a_U_unify, a_Lambda_unify] = DHINOffline(L_unify+gamma*M_unify, D_unify,k);
save ./data/dblp_nr/result/a_unify_embedding_nr.mat a_unify_embedding;

dapc = sum(W_apc,2);
D_apc = diag(dapc);
L_apc = D_apc- W_apc;  
W_apc = NormalizeAdj(W_apc,0,2);
M_apc = (eye(size(W_apc,1)) - W_apc)' * (eye(size(W_apc,1)) - W_apc);

[apc_embedding, U_apc, Lambda_apc] = DHINOffline(L_apc+gamma*M_apc, D_apc,k);
save ./data/dblp_nr/result/apc_embedding_nr.mat apc_embedding;

dpc = sum(W_pc,2);
D_pc = diag(dpc);
L_pc = D_pc - W_pc;  
W_pc = NormalizeAdj(W_pc,0,2);
M_pc = (eye(size(W_pc,1)) - W_pc)' * (eye(size(W_pc,1)) - W_pc);

[pc_embedding, U_pc, Lambda_pc] = DHINOffline(L_pc+gamma*M_pc, D_pc,k);
save ./data/dblp_nr/result/pc_embedding_nr.mat pc_embedding;

