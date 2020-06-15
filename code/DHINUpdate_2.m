function [Delta_eigenvalue,Delta_eigenvector,new_A] = DHINUpdate_2(old_eigenvalue, old_eigenvector, Delta_L, Delta_D,Delta_M, A, B, gamma)
        
    [n, d] = size(old_eigenvector);
    
    C = old_eigenvector'*Delta_L*old_eigenvector - diag(old_eigenvalue)*old_eigenvector'*Delta_D*old_eigenvector;

    % calculate delta lambda
    Delta_eigenvalue  = zeros(1,d);
    for i = 2:d
        Delta_eigenvalue(1,i) = C(i,i) + gamma * (A(:,i)'*B(:,i)+B(:,i)'*A(:,i));
    end
    
    % calculate alpha and delta u
    Alpha = zeros(d,d);
    Delta_eigenvector = zeros(n,d);
    for i = 2:d
        for p = 2:d
            if p ~= i
                Alpha(i,p) = (C(p,i) + gamma * (A(:,p)'*B(:,i)+B(:,p)'*A(:,i))) /  (old_eigenvalue(i) -old_eigenvalue (p));
            else
                Alpha(i,i) = 1;  % ???
            end
            Delta_eigenvector(:,i) = Delta_eigenvector(:,i) + Alpha(i,p) * old_eigenvector(:,p) ;
        end
    end

    % updata A
    new_A = zeros(n,d);
    for i = 2:d
        for j = 2:d
            new_A(:,i) = new_A(:,i) + Alpha (i,j)*(A(:,j)+B(:,j));
        end
    end
 
end