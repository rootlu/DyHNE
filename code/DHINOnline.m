function [embedding, new_eigenvector, new_eigenvalue, new_A] = DHINOnline(U, Lambda, Delta_L, Delta_D, Delta_M, A, B, k, gamma)

old_eigenvalue_tmp = diag(Lambda)';
old_eigenvalue = old_eigenvalue_tmp;
old_eigenvector = U;
[Delta_eigenvalue,Delta_eigenvector,new_A] = DHINUpdate(old_eigenvalue, old_eigenvector, Delta_L, Delta_D, Delta_M, A, B, gamma);
new_eigenvector = old_eigenvector + Delta_eigenvector;
new_eigenvalue  = old_eigenvalue + Delta_eigenvalue;

embedding = NormalizeFea(new_eigenvector(:,2:k+1));
