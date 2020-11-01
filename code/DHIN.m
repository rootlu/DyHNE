clc;clear;
addpath(genpath(pwd));

load ./data/dblp/aca.mat;
load ./data/dblp/apa.mat;
load ./data/dblp/ata.mat;
load ./data/dblp/pcp.mat;
load ./data/dblp/pap.mat;
load ./data/dblp/ptp.mat;

A_aca = aca;
A_apa = apa;
A_ata = ata;
A_pcp = pcp;
A_pap = pap;
A_ptp = ptp;

k = 100;

%% obtain diagonal and laplacian matrix

daca = sum(A_aca,2);
D_aca = diag(daca);
L_aca = D_aca - A_aca;  % TODO: normalize A_aca!!!
norm_A_aca = NormalizeAdj(A_aca);
M_aca = (eye(size(A_aca,1)) - A_aca)' * (eye(size(A_aca,1)) - A_aca);
% [V,D_eig] = eig(L_aca+M_aca);
% embedding = NormalizeFea(V(:,2:k+1));
% save ./data/dblp/result/12order_embedding.mat embedding;

[aca_embedding, V_aca, D_aca_eig] = DHINOffline(L_aca+M_aca,D_aca, k);
save ./data/dblp/result/aca_embedding.mat aca_embedding;

dapa = sum(A_apa,2);
D_apa = diag(dapa);
L_apa = D_apa - A_apa;
[apa_embedding, V_apa, D_apa_eig] = DHINOffline(L_apa, D_apa,k);
save ./data/dblp/result/apa_embedding.mat apa_embedding;

data = sum(A_ata,2);
D_ata = diag(data);
L_ata = D_ata - A_ata;
[ata_embedding, V_ata, D_ata_eig] = DHINOffline(L_ata, D_ata,k);
save ./data/dblp/result/ata_embedding.mat ata_embedding;

dpcp = sum(A_pcp,2);
D_pcp = diag(dpcp);
L_pcp = D_pcp - A_pcp;
[pcp_embedding, V_pcp, D_pcp_eig] = DHINOffline(L_pcp, D_pcp,k);
save ./data/dblp/result/pcp_embedding.mat pcp_embedding;

dpap = sum(A_pap,2);
D_pap = diag(dpap);
L_pap = D_pap - A_pap;
[pap_embedding, V_pap, D_pap_eig] = DHINOffline(L_pap, D_pap,k);
save ./data/dblp/result/pap_embedding.mat pap_embedding;

dptp = sum(A_ptp,2);
D_ptp = diag(dptp);
L_ptp = D_ptp - A_ptp;
[ptp_embedding, V_ptp, D_ptp_eig] = DHINOffline(L_ptp, D_ptp,k);
save ./data/dblp/result/ptp_embedding.mat ptp_embedding;

% %% evaluation w.r.t. node classification at time step t
% indices = crossvalind('Kfold',n,10);
% Accuracytmp = 0; F1macrotmp = 0; F1microtmp = 0;
% for m = 1:10
%     testidx = (indices == m); 
%     trainidx = ~testidx;
%     Xtrain = embedding(trainidx,:);
%     ytrain = y(trainidx,:);
%     Xtest  = embedding(testidx,:);
%     ytest  = y(testidx,:);
% 
%     model = train(ytrain, sparse(Xtrain), '-s 0 -q');
%     [predict_label, accuracy, decision_values] = predict(ytest, sparse(Xtest), model, '-q');
%     [micro, macro] = micro_macro_PR(predict_label,ytest);
%     Accuracytmp = Accuracytmp + accuracy(1);
%     F1macrotmp = F1macrotmp + macro.fscore;
%     F1microtmp = F1microtmp + micro.fscore;
% end
% Accuracy =  Accuracytmp/10;
% F1macro  = F1macrotmp/10;
% F1micro  = F1microtmp/10;
% 
% fprintf('Joint Accuracy: %f\n', Accuracy);
% fprintf('Joint F1macro: %f\n', F1macro);
% fprintf('Joint F1micro: %f\n', F1micro);

%% pertubate the data and obtain the new diagonal and laplacian matrix
% addratio = 0.001;
% removeratio = 0.001;
% 
% Aremove = removeedge(removeratio, A);
% Anew = Aremove;      
% danew = sum(Anew,2);
% Danew = diag(danew);
% Lanew = Danew - Anew;
% DeltaLa = Lanew - La;
% DeltaDa = Danew - Da;
% 
% Xadd = addcontent(addratio, X);
% Xnew = Xadd;      
% Sxnew = constructW(Xnew,options);
% Sxnew = full(Sxnew);      
% dxnew = sum(Sxnew,2);
% Dxnew = diag(dxnew);
% Lxnew = Dxnew - Sxnew;
% DeltaLx = Lxnew - Lx;
% DeltaDx = Dxnew - Dx;

% %% learn embedding at time step t+1
% embedding = DANE_Online(Va, Daeig, Vx, Dxeig, Vjoint, Djoint, DeltaLx, DeltaDx, DeltaLa, DeltaDa, l, k);
% 
% %% evaluation w.r.t. node classification at time step t+1
% indices = crossvalind('Kfold',n,10);
% Accuracytmp = 0; F1macrotmp = 0; F1microtmp = 0;
% for m = 1:10
%     testidx = (indices == m); 
%     trainidx = ~testidx;
%     Xtrain = embedding(trainidx,:);
%     ytrain = y(trainidx,:);
%     Xtest  = embedding(testidx,:);
%     ytest  = y(testidx,:);
% 
%     model = train(ytrain, sparse(Xtrain), '-s 0 -q');
%     [predict_label, accuracy, decision_values] = predict(ytest, sparse(Xtest), model, '-q');
%     [micro, macro] = micro_macro_PR(predict_label,ytest);
%     Accuracytmp = Accuracytmp + accuracy(1);
%     F1macrotmp = F1macrotmp + macro.fscore;
%     F1microtmp = F1microtmp + micro.fscore;
% end
% Accuracy =  Accuracytmp/10;
% F1macro  = F1macrotmp/10;
% F1micro  = F1microtmp/10;
% 
% fprintf('Joint Accuracy: %f\n', Accuracy);
% fprintf('Joint F1macro: %f\n', F1macro);
% fprintf('Joint F1micro: %f\n', F1micro);