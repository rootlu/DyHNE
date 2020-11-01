clc;clear;
addpath(genpath(pwd));

load ./data/dblp_lp/ap_csr_lp.mat;
% load ./data/dblp_lp/apc_csr_lp.mat;
% load ./data/dblp_lp/apt_csr_lp.mat;
load ./data/dblp_lp/apa_csr_lp.mat;
load ./data/dblp_lp/apcpa_csr_lp.mat;
load ./data/dblp_lp/aptpa_csr_lp.mat;

W_ap = ap_csr_lp;
% W_apc = apc_csr_lp;
% W_apt = apt_csr_lp;
W_apa = apa_csr_lp;
W_apcpa = apcpa_csr_lp;
W_aptpa = aptpa_csr_lp;


k = 100;
gamma = 1;

% obtain diagonal and laplacian matrix

% dap = sum(W_ap,2);
% D_ap = diag(dap);
% L_ap = D_ap - W_ap;  
% W_ap = NormalizeAdj(W_ap,0,2);
% M_ap = (eye(size(W_ap,1)) - W_ap)' * (eye(size(W_ap,1)) - W_ap);
% 
% [ap_embedding, U_ap, Lambda_ap] = DHINOffline(L_ap+gamma*M_ap, D_ap,k);
% save ./data/dblp/result/ap_embedding_lp.mat ap_embedding;
% 
% dapc = sum(W_apc,2);
% D_apc = diag(dapc);
% L_apc = D_apc- W_apc;  
% W_apc = NormalizeAdj(W_apc,0,2);
% M_apc = (eye(size(W_apc,1)) - W_apc)' * (eye(size(W_apc,1)) - W_apc);
% 
% [apc_embedding, U_apc, Lambda_apc] = DHINOffline(L_apc+gamma*M_apc, D_apc,k);
% save ./data/dblp/result/apc_embedding_lp.mat apc_embedding;
% 
% dapt = sum(W_apt,2);
% D_apt = diag(dapt);
% L_apt = D_apt - W_apt;  
% W_apt = NormalizeAdj(W_apt,0,2);
% M_apt = (eye(size(W_apt,1)) - W_apt)' * (eye(size(W_apt,1)) - W_apt);
% 
% [apt_embedding, U_apt, Lambda_apt] = DHINOffline(L_apt+gamma*M_apt, D_apt,k);
% save ./data/dblp/result/apt_embedding_lp.mat apt_embedding;

% dapa = sum(W_apa,2);
% D_apa = diag(dapa);
% L_apa = D_apa- W_apa;  
% W_apa = NormalizeAdj(W_apa,0,2);
% M_apa = (eye(size(W_apa,1)) - W_apa)' * (eye(size(W_apa,1)) - W_apa);
%  
% [apa_embedding, U_apa, Lambda_apa] = DHINOffline(L_apa+gamma*M_apa, D_apa,k);
% save ./data/dblp_lp/result/apa_embedding_lp.mat apa_embedding;
% 
% dapcpa = sum(W_apcpa,2);
% D_apcpa = diag(dapcpa);
% L_apcpa = D_apcpa- W_apcpa;  
% W_apcpa = NormalizeAdj(W_apcpa,0,2);
% M_apcpa = (eye(size(W_apcpa,1)) - W_apcpa)' * (eye(size(W_apcpa,1)) - W_apcpa);
% 
% [apcpa_embedding, U_apcpa, Lambda_apcpa] = DHINOffline(L_apcpa+gamma*M_apcpa, D_apcpa,k);
% save ./data/dblp_lp/result/apcpa_embedding_lp.mat apcpa_embedding;
% 
% daptpa = sum(W_aptpa,2);
% D_aptpa = diag(daptpa);
% L_aptpa = D_aptpa- W_aptpa;  
% W_aptpa = NormalizeAdj(W_aptpa,0,2);
% M_aptpa = (eye(size(W_aptpa,1)) - W_aptpa)' * (eye(size(W_aptpa,1)) - W_aptpa);
% 
% [aptpa_embedding, U_aptpa, Lambda_aptpa] = DHINOffline(L_aptpa+gamma*M_aptpa, D_aptpa,k);
% save ./data/dblp_lp/result/aptpa_embedding_lp.mat aptpa_embedding;
% 
% split_embedding = apa_embedding(1:14475,:)+aptpa_embedding(1:14475,:)+apcpa_embedding(1:14475,:);
% save ./data/dblp_lp/result/split_embedding_lp.mat split_embedding;

W_unify = W_apcpa+W_apa+W_aptpa;
%W_unify = W_ap;
dunify = sum(W_unify,2);
D_unify = diag(dunify);
L_unify = D_unify- W_unify;  
W_unify = NormalizeAdj(W_unify,0,2);

t1=clock;
M_unify = (eye(size(W_unify,1)) - W_unify)' * (eye(size(W_unify,1)) - W_unify);
t2=clock;
fprintf('Time for M_unify: %f s \n', etime(t2,t1));

[unify_embedding, U_unify, Lambda_unify] = DHINOffline(L_unify+gamma*M_unify, D_unify,k);
save ./data/dblp_lp/result/unify_embedding_lp.mat unify_embedding;




