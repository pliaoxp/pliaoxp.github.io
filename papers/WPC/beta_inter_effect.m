
% This code provides WPC estimator of beta
% One needs to input: 
%%% X:  T by d by N, where T is dimension of time, d is dimension of beta,
%%%     N is dimension of cross sections
%%% Y:  N by T response
%%% r:  number of factors. One can consistently estimate it using the
%%%     outlined in Bai 09 (appendix). Alternatively, as Moon and Weidner
%%%     2013 showed, try a relatively large r.
%%% W:  N by N.  
%%%    Optional. The weight matrix W is used for WPC. The default choice is the
%%%    estimated inverse residual covariance matrix, which is the
%%%    first-order asymptotically optimal. Users can re-define W.

% Outputs are:
%%%  betaBai: The estimator using the method of Bai (2009)
%%%  betaWPC: The WPC estimator using weight matrix W (default is inverse SigmaU)
%%%  standard: the standard error of betaWPC. 



mm=size(X);  
d=mm(2); T=mm(1); N=mm(3);


 
  

for t=1:T
    for i=1:N 
       X2(i,:,t)=X(t,:,i);     % 1 x d,  X2(::t) is N x d
       XnoUseNew(i,t,:)=X(t,:,i);
    end;
end;
 

 %%%%%%%%%%% calculation of Bai 09
    
    M=zeros(d,d);
    for t=1:T
        M=M+X2(:,:,t)'*X2(:,:,t);
    end;
   
    Delta=10;
    betahat=zeros(d,1);
    
     
    while Delta>10^(-10)
        beta0=betahat;
        
        for i=1:N
            Z(i,:)=Y(i,:)-beta0'*X(:,:,i)'; % 1 x T;  Z is N x T
        end;
        [F,L]=factor(Z,eye(N),r);  % L is N by r
        G=zeros(d,1);
        for t=1:T
            G=G+X2(:,:,t)'*(Y(:,t)-L*F(t,:)');
        end;
        betahat=M\G; % inv(matrix)*G
        Delta=norm(betahat-beta0,'fro');
    end;
   betaBai=betahat;
  
     
    %%%%%%%%%%%% WPC  Bai and Liao (2012)
    hatu=zeros(N,T);
    for i=1:N
        hatu(i,:)=Y(i,:)-betaBai'*X(:,:,i)'-L(i,:)*F';   % 1 x T   hatu is N x T
    end;
    Sigu=matrix(hatu,1);  % can also replace 1 with a slightly smaller value (e.g., 0.5, 0.7)
    Delta=10;
    betahat=betaBai;
    M2=zeros(d,d);
    
     
    
    for t=1:T
        M2=M2+X2(:,:,t)'*(Sigu\X2(:,:,t)); 
    end;
    Z=zeros(N,T);
    while Delta>10^(-10)
        beta0=betahat;
        for i=1:N
            Z(i,:)=Y(i,:)-beta0'*X(:,:,i)'; % 1 x T;  Z is N x T
        end;
        [F,L]=factor(Z,Sigu,r);
        G=zeros(d,1);
        for t=1:T
            G=G+X2(:,:,t)'*(Sigu\(Y(:,t)-L*F(t,:)'));   % N x 1
        end;
        betahat=M2\G;
        Delta=norm(betahat-beta0,'fro');
    end;
     betaWPC=betahat;
       Mf=eye(T)-(F/(F'*F))*F'/T;
     
     %%% Estimate standard error of betaWPC

  W=inv(Sigu);
  
 A1=W-W*(L/(L'*W*L))*L'*W;  
 AF=kron(A1,Mf);     
 ZZ=zeros(N*T,d);
 for k=1:d
     ZZ(:,k)=reshape(XnoUseNew(:,:,k)',N*T,1);    % ZZ is NT by d
 end;
  DF=ZZ'*AF*ZZ/(N*T);
  % ErrVar is the estimated Sigma_u,  Bai 2009 estimates it as diagonal
    ErrVar=Sigu;
  varia=(DF\(ZZ'))*AF*(kron(ErrVar,eye(T)))*AF*(ZZ/DF)/(N^2*T^2);
   standard=diag(varia).^(1/2); % standard error
      