clc;clear;
addpath(genpath(pwd));

t1=clock;
load ./data/dblpDynamic/apa_csr_0.mat;
load ./data/dblpDynamic/apcpa_csr_0.mat;
load ./data/dblpDynamic/aptpa_csr_0.mat;

W_apa = apa_csr_0;
W_apcpa = apcpa_csr_0;
W_aptpa = aptpa_csr_0;

k = 100;
gamma = 1;

W= W_apcpa+0.1*W_apa+0.9*W_aptpa;
d = sum(W,2);
D = diag(d);
L = D- W;  
W_norm = NormalizeAdj(W,0,2);
M = (eye(size(W_norm,1)) - W_norm)' * (eye(size(W_norm,1)) - W_norm);

[embedding, U, Lambda] = DHINOffline(L+gamma*M , D,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/dblpDynamic/result/static_0.1apa+1apcpa+0.9aptpa_embedding.mat embedding;


%% pertubate the data and obtain the new diagonal and laplacian matrix
A = (W_norm-eye(size(W_norm,1)))*U;
% B = zeros(size(W_norm,1),size(U,2));
for i = 1:9
    fprintf('Time step: %d  \n', i) 
    apa_data = ['./data/dblpDynamic/apa_csr_',int2str(i),'.mat'];
    apcpa_data = ['./data/dblpDynamic/apcpa_csr_',int2str(i),'.mat'];
    aptpa_data = ['./data/dblpDynamic/aptpa_csr_',int2str(i),'.mat'];
    W_apa = cell2mat(struct2cell(load(apa_data)));
    W_apcpa = cell2mat(struct2cell(load(apcpa_data)));
    W_aptpa = cell2mat(struct2cell(load(aptpa_data)));
    
    W_new = W_apcpa+0.1*W_apa+0.9*W_aptpa;
    d_new = sum(W_new,2);
    D_new = diag(d_new);
    L_new = D_new - W_new;  
	W_new_norm = NormalizeAdj(W_new,0,2);
% 	M_new = (speye(size(W_new_norm,1)) - W_new_norm)' * (speye(size(W_new_norm,1)) - W_new_norm);
    
    t1=clock;
    Delta_L = L_new - L;
    Delta_D = D_new - D;
    Delta_W = W_new_norm - W_norm;
    Delta_M = 0;
%     Delta_M  =  M_new - M;
%     Delta_M = (W_norm-eye(size(W_norm,1)))'*Delta_W+Delta_W'*(W_norm-eye(size(W_norm,1)));
    
    B = Delta_W*U;  

    %% learn embedding at time step t+1
    [embedding, U_new, Lambda_new, A_new]= DHINOnline(U, Lambda, Delta_L, Delta_D, Delta_M, A, B, k, gamma);
  
    t2=clock;
    fprintf('Time for one dynamic update: %f s  \n', etime(t2,t1)) 
    
    current_time_step_embs =  ['./data/dblpDynamic/result/',int2str(i),'_0.1apa+1apcpa+0.9aptpa_embedding.mat'];
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

save ./data/dblpDynamic/result/final_0.1apa+1apcpa+0.9aptpa_embedding.mat embedding;
