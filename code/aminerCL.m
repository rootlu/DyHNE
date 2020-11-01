clc;clear;
addpath(genpath(pwd));

% load ./data/aminer/ap_csr.mat;
% load ./data/aminer/apc_csr.mat;
load ./data/aminer/apa_csr.mat;
load ./data/aminer/apcpa_csr.mat;
load ./data/aminer/aptpa_csr.mat;

% W_ap = ap;
% W_apc = apc_csr;
W_apa = apa_csr;
W_apcpa = apcpa_csr;
W_aptpa = aptpa_csr;

k = 100;
gamma = 1;

% obtain diagonal and laplacian matrix

% dap = sum(W_ap,2);
% D_ap = diag(dap);
% L_ap = D_ap - W_ap;  
% W_ap = NormalizeAdj(W_ap);
% M_ap = (eye(size(W_ap,1)) - W_ap)' * (eye(size(W_ap,1)) - W_ap);
% 
% [ap_embedding, U_ap, Lambda_ap] = DHINOffline(L_ap+gamma*M_ap, D_ap,k);
% save ./data/aminer/result/ap_embedding.mat ap_embedding;

% dapc = sum(W_apc,2);
% D_apc = diag(dapc);
% L_apc = D_apc- W_apc;  
% W_apc = NormalizeAdj(W_apc,0,2);
% M_apc = (eye(size(W_apc,1)) - W_apc)' * (eye(size(W_apc,1)) - W_apc);
% 
% [apc_embedding, U_apc, Lambda_apc] = DHINOffline(L_apc+gamma*M_apc, D_apc,k);
% save ./data/aminer/result/apc_embedding.mat apc_embedding;

% dapa = sum(W_apa,2);
% D_apa = diag(dapa);
% L_apa = D_apa- W_apa;  
% W_apa = NormalizeAdj(W_apa,0,2);
% M_apa = (eye(size(W_apa,1)) - W_apa)' * (eye(size(W_apa,1)) - W_apa);
%  
% [apa_embedding, U_apa, Lambda_apa] = DHINOffline(L_apa+gamma*M_apa, D_apa,k);
% save ./data/aminer/result/apa_embedding.mat apa_embedding;
% 
% dapcpa = sum(W_apcpa,2);
% D_apcpa = diag(dapcpa);
% L_apcpa = D_apcpa- W_apcpa;  
% W_apcpa = NormalizeAdj(W_apcpa,0,2);
% M_apcpa = (eye(size(W_apcpa,1)) - W_apcpa)' * (eye(size(W_apcpa,1)) - W_apcpa);
% 
% [apcpa_embedding, U_apcpa, Lambda_apcpa] = DHINOffline(L_apcpa+gamma*M_apcpa, D_apcpa,k);
% save ./data/aminer/result/apcpa_embedding.mat apcpa_embedding;

% split_embedding = 0.1*apa_embedding(1:22942,:)+apcpa_embedding(1:22942,:);
% save ./data/aminer/result/split_apcpa+0.1apa_embedding.mat split_embedding;

t1=clock;
W_unify = 0.1*W_apa+W_apcpa+0.9*W_aptpa;
dunify = sum(W_unify,2);
D_unify = diag(dunify);
L_unify = D_unify - W_unify;  
W_unify = NormalizeAdj(W_unify,0,2);
H = speye(size(W_unify,1)) - W_unify;
clear W_apa W_apcpa W_aptpa W_unify
M_unify = H'*H;
X = L_unify+gamma * M_unify;
% save ./data/aminer/X.mat X;
[unify_embedding, U_unify, Lambda_unify] = DHINOffline(X, D_unify,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/aminer/result/unify_0.1apa+apcpa+0.9aptpa_embedding.mat unify_embedding;
