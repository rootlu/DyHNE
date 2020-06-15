% function A = NormalizeAdj(A,row)
% % if row == 1, normalize each row of A to have unit norm;
% % if row == 0, normalize each column of A to have unit norm;
% 
% 
% if ~exist('row','var')
%     row = 1;
% end
% 
% if row
% %     A = A./repmat(sqrt(sum(A.^2,2)),1,size(A,2));
%     A = A./repmat(sum(A,2),1,size(A,2));
% 
% else
% %     A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);
%     A =  A./repmat(sum(A,1),size(A,1),1);
% end
%             
% return;

function [normMat] = NormalizeAdj(oriMat, isSqrt, type)
%normlized  the matrix in terms of row or col
%oriMat is the original matrix
%isSqrt means whether sqrt normlization
%type is the normalized type. 1 is the column normalization; 2 is the row
%normalization.
%normMat is the normalized matrix.

if(type == 1)
%normalization according to column
    sumVect = sum(oriMat,1);
    if(isSqrt)
        sumVect = power(sumVect,0.5);
    end
    sumVect = 1./sumVect;
    len = length(sumVect);
    l = linspace(1,len,len);
    diagMat = spconvert([l',l',sumVect']);
    normMat = oriMat * diagMat;
else
 %normalization according to row
    sumVect = sum(oriMat,2);
    if(isSqrt)
        sumVect = power(sumVect,0.5);
    end
    sumVect = 1./sumVect;
    len = length(sumVect);
    l = linspace(1,len,len);
    diagMat = spconvert([l',l',sumVect]);
    normMat = diagMat * oriMat;
end

end

