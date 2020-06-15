clc;clear;
addpath(genpath(pwd));

t1=clock;
load ./data/dblpDynamic_lp_cikm/apa_csr_0.mat;
load ./data/dblpDynamic_lp_cikm/apcpa_csr_0.mat;
load ./data/dblpDynamic_lp_cikm/aptpa_csr_0.mat;

W_apa = apa_csr_0;
W_apcpa = apcpa_csr_0;
W_aptpa = aptpa_csr_0;

k = 100;
gamma = 1;

W= W_apa+W_apcpa+W_aptpa;
d = sum(W,2);
D = diag(d);
L = D- W;  

W_norm = NormalizeAdj(W,0,2);
clear W_apa W_apcpa W_aptpa apa_csr_0 apcpa_csr_0 aptpa_csr_0;
H = eye(size(W_norm,1)) - W_norm;

M = H'*H;
X = L+gamma * M;
clear M H;
[embedding, U, Lambda] = DHINOffline(X, D,k);

t2=clock;
fprintf('Time for static model: %f s  \n', etime(t2,t1)) 

save ./data/dblpDynamic_lp_cikm/result/0_apa_embedding.mat embedding;


%% pertubate the data and obtain the new diagonal and laplacian matrix
A = (W_norm-eye(size(W_norm,1)))*U;
for i = 1:10
    fprintf('Time step: %d  \n', i) 
    apa_data = ['./data/dblpDynamic_lp_cikm/apa_csr_',int2str(i),'.mat'];
    apcpa_data = ['./data/dblpDynamic_lp_cikm/apcpa_csr_',int2str(i),'.mat'];
    aptpa_data = ['./data/dblpDynamic_lp_cikm/aptpa_csr_',int2str(i),'.mat'];
    W_apa = cell2mat(struct2cell(load(apa_data)));
    W_apcpa = cell2mat(struct2cell(load(apcpa_data)));
    W_aptpa = cell2mat(struct2cell(load(aptpa_data)));
    
    W_new = W_apa+W_apcpa+W_aptpa;
    W_new_norm = NormalizeAdj(W_new,0,2);
    
    clear W_apa W_apcpa W_aptpa;

    d_new = sum(W_new,2);
    D_new = diag(d_new);
    L_new = D_new - W_new;  
    
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
    
    current_time_step_embs =  ['./data/dblpDynamic_lp_cikm/result/',int2str(i),'_apa+apcpa+aptpa_embedding.mat'];
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
