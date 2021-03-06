function [Delta_eigenvalue,Delta_eigenvector,new_A] = DHINUpdate(old_eigenvalue, old_eigenvector, Delta_L, Delta_D, Delta_M, A, B, gamma)
    % Input oldeigenvalue is stored in a one-by-d diagonal matrix
    % Input oldeigenvector is stored in a n-by-d matrix
    % Input DeltaL denotes the change of the laplacian matrix
    % Input DeltaD denotes the change of the diagonal matrix
        
    [n, d] = size(old_eigenvector);
    Delta_eigenvalue  = zeros(1,d);
    Delta_eigenvector = zeros(n,d);
  
    alpha = zeros(d,d);
    new_A = zeros(n,d);
    
    H = (A+B)'*B+B'*(A+B);   % d x d 
    G = old_eigenvector'*Delta_L*old_eigenvector - diag(old_eigenvalue)*old_eigenvector'*Delta_D*old_eigenvector;
    
%     for i = 2:d
%         old_eigenvector_i = old_eigenvector(:,i);
%         old_eigenvalue_i  = old_eigenvalue(1,i);
%         
%         %% compute the change of eigenvalue
%         Delta_eigenvalue(1,i) = old_eigenvector_i'*Delta_L*old_eigenvector_i - old_eigenvalue_i*old_eigenvector_i'*Delta_D*old_eigenvector_i;
%         
%         %% compute the change of eigenvector
%         for j = 2:d
%             old_eigenvector_j = old_eigenvector(:,j);
%             old_eigenvalue_j  = old_eigenvalue(1,j);
%             if j ~= i
%                 Delta_eigenvector(:,i) = Delta_eigenvector(:,i) + (old_eigenvector_j'*Delta_L*old_eigenvector_i-old_eigenvalue_i*old_eigenvector_j'*Delta_D*old_eigenvector_i)*old_eigenvector_j/(old_eigenvalue_i-old_eigenvalue_j);
%             end
%         end
%     end
    
    for i = 2:d
        old_eigenvector_i = old_eigenvector(:,i);  % n x d
        old_eigenvalue_i  = old_eigenvalue(1,i);  % n x 1
        
        %% compute the change of eigenvalue
        Delta_eigenvalue(1,i) = G(i,i) + gamma*H(i,i);
%         Delta_eigenvalue(1,i) = (old_eigenvector_i'*Delta_L*old_eigenvector_i - old_eigenvalue_i*old_eigenvector_i'*Delta_D*old_eigenvector_i + gamma*old_eigenvector_i'*Delta_M*old_eigenvector_i);

        %% compute the change of eigenvector
        for j = 2:d
            old_eigenvector_j = old_eigenvector(:,j);
            old_eigenvalue_j  = old_eigenvalue(1,j);
            if j ~= i
                alpha(i,j) = (G(j,i)+ gamma*H(j,i)) / (old_eigenvalue_i-old_eigenvalue_j);
%                 Delta_eigenvector(:,i) = Delta_eigenvector(:,i) + (old_eigenvector_j'*Delta_L*old_eigenvector_i - old_eigenvalue_i*old_eigenvector_j'*Delta_D*old_eigenvector_i + gamma*old_eigenvector_j'*Delta_M*old_eigenvector_i)*old_eigenvector_j / (old_eigenvalue_i-old_eigenvalue_j);
%             else
%                  alpha(i,j) = -0.5*old_eigenvector_i'*Delta_D*old_eigenvector_i;
            end
            Delta_eigenvector(:,i) = Delta_eigenvector(:,i) + alpha(i,j) * old_eigenvector_j ;

        end
    end
    
%     for i = 2:d
%             for z = 2:d
%                 if z ~=i
%                     new_A(:,i) = new_A(:,i) + alpha(i,z) * (A(:,z) + B(:,z));
%                 end
%             end
%     end

end