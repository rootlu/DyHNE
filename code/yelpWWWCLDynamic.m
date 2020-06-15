clc;clear;
addpath(genpath(pwd));

t1=clock;
bsb_data = ['./data/yelpWWWDynamic/bsb_csr_0.mat'];
brurb_data = ['./data/yelpWWWDynamic/brurb_csr_0.mat'];
W_bsb = cell2mat(struct2cell(load(bsb_data)));
W_brurb = cell2mat(struct2cell(load(brurb_data)));

k = 100;
gamma = 1;

W= 0.4*W_bsb+0.6*W_brurb;
d = sum(W,2);
D = diag(d);
L = D - W;  
W_norm = NormalizeAdj(W,0,2);
M = (eye(size(W_norm,1)) - W_norm)' * (eye(size(W_norm,1)) - W_norm);

[embedding, U, Lambda] = DHINOffline(L+M , D,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/yelpWWWDynamic/result/0_0.4bsb+0.6brurb_embedding.mat embedding;


%% pertubate the data and obtain the new diagonal and laplacian matrix
A = (W_norm-eye(size(W_norm,1)))*U;
for i = 1:10
    fprintf('Time step: %d  \n', i) 
    bsb_data = ['./data/yelpWWWDynamic/bsb_csr_',int2str(i),'.mat'];
    brurb_data = ['./data/yelpWWWDynamic/brurb_csr_',int2str(i),'.mat'];
    W_bsb = cell2mat(struct2cell(load(bsb_data)));
    W_brurb = cell2mat(struct2cell(load(brurb_data)));
    
    W_new = 0.4*W_bsb+0.6*W_brurb;
    d_new = sum(W_new,2);
    D_new = diag(d_new);
    L_new = D_new - W_new;  
	W_new_norm = NormalizeAdj(W_new,0,2);
    
    t1=clock;
    Delta_L = L_new - L;
    Delta_D = D_new - D;
    Delta_W = W_new_norm - W_norm;
    Delta_M = 0;
    
    B = Delta_W*U;  

    %% learn embedding at time step t+1
    [embedding, U_new, Lambda_new, A_new]= DHINOnline(U, Lambda, Delta_L, Delta_D, Delta_M, A, B, k, gamma);
  
    t2=clock;
    fprintf('Time for one dynamic update: %f s  \n', etime(t2,t1)) 
    
    current_time_step_embs =  ['./data/yelpWWWDynamic/result/',int2str(i),'_0.4bsb+0.6brurb_embedding.mat'];
    save (current_time_step_embs, 'embedding');

%     L = L_new;  
%     D= D_new;
% 	M= M_new;
%     W = W_new;
%     U = U_new;
%     Lambda = diag(Lambda_new);
%     A = A_new;
%     B = B_new;
end
