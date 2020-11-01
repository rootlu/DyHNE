clc;clear;
addpath(genpath(pwd));

load ./data/aminer_lp/apa_csr_lp.mat;
load ./data/aminer_lp/apcpa_csr_lp.mat;
load ./data/aminer_lp/aptpa_csr_lp.mat;

W_apa = apa_csr_lp;
W_apcpa = apcpa_csr_lp;
W_aptpa = aptpa_csr_lp;

k = 100;
gamma = 1;

t1=clock;
W_unify = W_apa;
dunify = sum(W_unify,2);
D_unify = diag(dunify);
L_unify = D_unify- W_unify;  
W_unify= NormalizeAdj(W_unify,0,2);
clear W_apa W_apcpa W_aptpa apa_csr apcpa_csr aptpa_csr;
H = eye(size(W_unify,1)) - W_unify;
clear W_unify
M_unify = H'*H;
X = L_unify+M_unify;
clear M_unify H;
[unify_embedding, U_unify, Lambda_unify] = DHINOffline(X, D_unify,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/aminer_lp/result/unify_apa_embedding_lp.mat unify_embedding;




