function [embedding, U, Lambda] = DHINOffline(L, D, d)

n = size(D,1);

epsilon = 1e-6;

opts.v0 = rand(n,1);
% opts.tol = 1e-3;
% t1=clock;
% [U,Lambda] = eigs(L+epsilon*eye(n),D+epsilon*eye(n),d+1,'smallestreal');
% [U,Lambda] = eigs(L+epsilon*eye(n),D+epsilon*eye(n),d+1,'SR');
[U,Lambda] = eig(L+epsilon*eye(n),D+epsilon*eye(n));
% t2=clock
% fprintf('Time for eigs: %f s  \n', etime(t2,t1)) 

embedding = NormalizeFea(U(:,2:d+1));

U = U(:,1:d+1);
Lambda = Lambda(:,1:d+1);

end