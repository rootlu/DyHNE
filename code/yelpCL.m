clc;clear;
addpath(genpath(pwd));

load ./data/yelp/bsb_csr.mat;
load ./data/yelp/bcb_csr.mat;
load ./data/yelp/brurb_csr.mat;

W_bsb = bsb_csr;
W_bcb = bcb_csr;
W_brurb = brurb_csr;

k = 100;
gamma = 1;

% obtain diagonal and laplacian matrix
t1=clock;
W_unify = W_bsb+W_bcb+W_brurb;
dunify = sum(W_unify,2);
D_unify = diag(dunify);
L_unify = D_unify- W_unify;  
W_unify = NormalizeAdj(W_unify,0,2);
M_unify = (speye(size(W_unify,1)) - W_unify)' * (speye(size(W_unify,1)) - W_unify);

[unify_embedding, U_unify, Lambda_unify] = DHINOffline(0.5*L_unify + 0.5*M_unify, D_unify,k);
save ./data/yelp/result/unify_0.1bsb+0.1bcb+0.8brurb_embedding_0.5g0.5.mat unify_embedding;

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 
    