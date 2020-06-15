clc;clear;
addpath(genpath(pwd));

t1=clock;

apa_data = './data/aminerDynamic_lp/apa_csr_2003.mat';
apcpa_data = './data/aminerDynamic_lp/apcpa_csr_2003.mat';
aptpa_data = './data/aminerDynamic_lp/aptpa_csr_2003.mat';
W_apa = cell2mat(struct2cell(load(apa_data)));
W_apcpa = cell2mat(struct2cell(load(apcpa_data)));
W_aptpa = cell2mat(struct2cell(load(aptpa_data)));

k = 100;
gamma = 1;

t1=clock;
W_static = W_apa+W_apcpa+W_aptpa;
d_static = sum(W_static,2);
D_static = diag(d_static);
L_static = D_static - W_static;  
W_static_norm = NormalizeAdj(W_static,0,2);
clear W_apa W_apcpa W_aptpa apa_csr apcpa_csr aptpa_csr;

H = eye(size(W_static_norm,1)) - W_static_norm;
M_static = H'*H;
X = L_static+gamma * M_static;
clear M_static H;
[embedding_static, U, Lambda] = DHINOffline(X, D_static,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/aminerDynamic_lp/result/2003_apa+apcpa+aptpa_embedding_lp.mat embedding_static;


%% pertubate the data and obtain the new diagonal and laplacian matrix
A = (W_static_norm-eye(size(W_static_norm,1)))*U;

for i = 1:10
    fprintf('Time step: %d  \n', i) 
    apa_data = ['./data/aminerDynamic_lp/apa_csr_2004_',int2str(i),'.mat'];
    apcpa_data = ['./data/aminerDynamic_lp/apcpa_csr_2004_',int2str(i),'.mat'];
    aptpa_data = ['./data/aminerDynamic_lp/aptpa_csr_2004_',int2str(i),'.mat'];
    W_apa = cell2mat(struct2cell(load(apa_data)));
    W_apcpa = cell2mat(struct2cell(load(apcpa_data)));
    W_aptpa = cell2mat(struct2cell(load(aptpa_data)));
    
    W_new = W_apa+W_apcpa+W_aptpa;
    d_new = sum(W_new,2);
    D_new = diag(d_new);
    L_new = D_new - W_new;  
	W_new_norm = NormalizeAdj(W_new,0,2);
    clear W_apa W_apcpa W_aptpa;

    t1=clock;
    Delta_L = L_new - L_static;
    Delta_D = D_new - D_static;
    Delta_W = W_new - W_static;
    Delta_W_norm = W_new_norm - W_static_norm;
    Delta_M = 0;
    
    B = Delta_W_norm*U;  

    %% learn embedding at time step t+1
    [embedding, U_new, Lambda_new, A_new]= DHINOnline(U, Lambda, Delta_L, Delta_D, Delta_M, A, B, k, gamma);
  
    t2=clock;
    fprintf('Time for one dynamic update: %f s  \n', etime(t2,t1)) 
    
    current_time_step_embs =  ['./data/aminerDynamic_lp/result/2004_',int2str(i),'_apa+apcpa+aptpa_embedding_lp.mat'];
    save (current_time_step_embs, 'embedding');
    
%     U = U_new;
%     Lambda = diag(Lambda_new);
%     A = A_new;
%     
%     L_static = L_new;
%     D_static = D_new;
%     W_norm_static =  W_new_norm;

end