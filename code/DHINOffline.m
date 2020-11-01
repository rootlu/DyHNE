function [embedding, U, Lambda] = DHINOffline(L, D, k)
%% Input
% Dx -- diagonal matrix of the attribute similarity matrix
% Lx -- laplacian matrix of the attribute similarity matrix
% Da -- diagonal matrix of the network adjacency matrix
% La -- laplacian matrix of the network adjacency matrix
% k -- intermediate embedding dimension
% l -- final embedding dimension


n = size(D,1);

epsilon = 1e-6;

opts.v0 = rand(n,1);
% opts.tol = 1e-3;
% t1=clock;
[U,Lambda] = eigs(L+epsilon*eye(n),D+epsilon*eye(n),k+1,'smallestreal');
% [U,Lambda] = eigs(L,D,k+1,'smallestreal',opts);

% [U,Lambda] = eig(L+epsilon*eye(n),D+epsilon*eye(n));

% L = single(L+epsilon*eye(n));
% D = single(D+epsilon*eye(n));
% L = gpuArray(L);
% D = gpuArray(D);
% U = gather(U);
% Lambda = gather(Lambda);

% t2=clock
% fprintf('Time for eigs: %f s  \n', etime(t2,t1)) 

embedding = NormalizeFea(U(:,2:k+1));
U = U(:,1:k+1);
Lambda = Lambda(:,1:k+1);

end