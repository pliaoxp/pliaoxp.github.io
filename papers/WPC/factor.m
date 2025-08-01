function [F,L]=factor(Y,S,r)
% returns estimated factors and loadings using the weight matrix S^{-1}
% When S=identity matrix, returns regular PCA estimators as Bai (2003)
% Inputs:  Y: N by T data matrix
%          S: N by N weight matrix, in fact the weight we use is S^{-1}
%             The optimal S should be close to the error covariance matrix.
%          r: number of factors; if unknown, can be consistently estimated
% Outputs: F: T by r, common factors. Each row represents each observation
%              of the r factors
%          L: N by r, factor loadings

[N,T]=size(Y);
Y= Y-mean(Y')'*ones(1,T) ;  %de-mean
[V,Dd]=eig(Y'*(S\Y));  % Y'*inv(S)*Y
[W,Id]=sort(diag(Dd));
for i=1:r
        F(:,i)=sqrt(T)*V(:,Id(T-i+1));   
end;
L=Y*F/T;  

