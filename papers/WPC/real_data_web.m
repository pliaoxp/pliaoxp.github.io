

% only additive effects, no time trend
clear;

%load divrate
load ko;
%%%%  USED IN THE PAPER, ALSO contains choosing number of factor=10 
%%%%  it seems if threshold large so that SigmaU is diagonal,
%%%%  insignificant, which means the inefficiency is indeed due to missing
%%%%  cross-sectional correlations.

% %%%%%%%%%%%%%%%%%%  the 991-1023 rows of ko are outliers, this code removes it
%konew=[ko(1:990,:);ko(1024:1584,:)];
%clear ko;
%ko=konew;
%clear konew;


%ko(529,2)=ko(530,2);
%ko(529,3)=ko(530,3);


% We make this change because the "adjusted divorce rate" from KO was -0.0704; However, divorce rate cannot be negative. 
% So we replace it with the following year's divorce rate.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BigX=[ko(:,4:11)]; % include 8 treatmeht
T=33;
N=length(ko(:,1))/T;  %48 is raw data, 47 is after the 991-1023 rows are removed ;


 r=10; % 10 number of factors
 
 
 J=3;% J=2:  raw divorce rate, before log;  J=3, log divorce rate after taking log  

 
% cr=1.96;  % 1.96%       2.58 is critical value for 99% confidence interval

 
 
 
for i=1:N
    for t=1:T
         yold(i,t)=ko((i-1)*T+t,J);   % N by T    
    end;
end

 
for k=1:8  
    for i=1:N
        for t=1:T
             X2old(i,k,t)=BigX((i-1)*T+t,k);  %X2(::t) is N x 8, If just linear trend, 1:9
             Xold(t,k,i)=BigX((i-1)*T+t,k);   % X(:,:,i) is T by 8,  If just linear trend, 1:9
             XnoUse(i,t,k)=BigX((i-1)*T+t,k);   % XnoUse(::k) is N by T but no use  later
        end;
    end;
end;


for k=1:8   
    for i=1:N
        for t=1:T
            value=X2old(i,k,t)-mean(XnoUse(i,:,k))-mean(XnoUse(:,t,k))+mean(BigX(:,k));
             X2(i,k,t)= value;  %X2(::t) is N x 8, If just linear trend, 1:9
              X(t,k,i)=value; % X(:,:,i) is T by 8,  If just linear trend, 1:9
             XnoUseNew(i,t,k)=value;   % XnoUse(::k) is N by T but no use  later 
             clear value;
        end;
    end;
end;
for i=1:N
    for t=1:T
        y(i,t)=yold(i,t)-mean(yold(i,:))-mean(yold(:,t))+mean(ko(:,J));
    end;
end;
M=zeros(8,8);
for t=1:T
    M=M+X2(:,:,t)'*X2(:,:,t);  % used in Bai
end; 
 


 
     %%%% Data: y, X2 X
 %%%%%%%%%%% alternative initial value of Bai 09
  clear Delta F L G betahat beta0 ZZ betaBai  

 Delta=10;
 F=zeros(T,r);
 L=zeros(N,r);
 G=zeros(8,1);
 for t=1:T
      G=G+X2(:,:,t)'*(y(:,t)-L*F(t,:)');
 end
 betahat=M\G;
 while Delta>10^(-10)
     beta0=betahat;
     for i=1:N
         ZZ(i,:)=y(i,:)-beta0'*X(:,:,i)'; % 1 x T;  Z is N x T
     end;
     [F,L]=factor(ZZ,eye(N),r); % L is N by r  F is T by r
     G=zeros(8,1);
     for t=1:T
         G=G+X2(:,:,t)'*(y(:,t)-L*F(t,:)');
     end;
     betahat=M\G; % inv(matrix)*G
     Delta=norm(betahat-beta0,'fro');
 end;
 betabai=betahat ;  % faster
    
 

 
% estimate residual N by T**************************
 for t=1:T
     u(:,t)=y(:,t)- X2(:,:,t)*betabai-L*F(t,:)';            %  u(:t) is N by 1    X2(::t) is N x 8,
 end;
 % residual sample covariance
 Su=u*u'/(T-2);
 Sigu=matrix(u,.51); 
 
 
 
  
 
 % *estimate asym variance  Bai 2009  ************************************
 W=eye(N); % W   the weight matrix, can be inv SigmaU
 Mf=eye(T)-(F/(F'*F))*F'/T;
 A1=W-W*(L/(L'*W*L))*L'*W;  
 AF=kron(A1,Mf);     
 
 for k=1:8

     Z(:,k)=reshape(XnoUseNew(:,:,k)',N*T,1);    % Z is NT by 8
 end;
  DF=Z'*AF*Z/(N*T);
  % ErrVar is the estimated Sigma_u,  Bai 2009 estimates it as diagonal
  ErrVar=diag(diag(Su));
  varBai=(DF\(Z'))*AF*(kron(ErrVar,eye(T)))*AF*(Z/DF)/(N^2*T^2);
   standard=diag(varBai).^(1/2); % standard error of betaBai
     for i=1:8  % confidence intervals
         conbai96(i,:)=[betabai(i)-1.96*standard(i), betabai(i)+1.96*standard(i)];
         conbai58(i,:)=[betabai(i)-2.58*standard(i), betabai(i)+2.58*standard(i)];
        
     end;
 
 
 varbai=diag(varBai);
 
 
     
 %%%%%%%%%%% Estimate beta using WPC Liao
  clear       beta0 ZZ  
    
   M2=zeros(8,8);
    for t=1:T
        M2=M2+X2(:,:,t)'*(Sigu\X2(:,:,t));  % X2(:,:,t)  is N x d
    end;
     Delta=10;
    while Delta>10^(-10)
        beta0=betahat;
        for i=1:N
            ZZ(i,:)=y(i,:)-beta0'*X(:,:,i)'; % 1 x T;  Z is N x T
        end;
        [F,L]=factor(ZZ,Sigu,r);
        G=zeros(8,1);
        for t=1:T
            G=G+X2(:,:,t)'*(Sigu\(y(:,t)-L*F(t,:)'));   % N x 1
        end;
        betahat=M2\G;
        Delta=norm(betahat-beta0,'fro');
    end;
     betaliao=betahat;
     Mf=eye(T)-(F/(F'*F))*F'/T;

 
 
 
 
 % *estimate asym variance  Liao 2012  ************************************
 W=Sigu; % W   the Inverse of W is the weight matrix, so can be W= SigmaU, then the weight is inv SigmaU
clear standard;
 A1=inv(W)-W\((L/(L'*(W\L)))*(L'/W));   % Here W= Sigma U

 AF=kron(A1,Mf);     
 
 for k=1:8

     Z(:,k)=reshape(XnoUseNew(:,:,k)',N*T,1);    % Z is NT by 8
 end;
  DF=Z'*AF*Z/(N*T);
  % ErrVar is the estimated Sigma_u,  Bai 2009 estimates it as diagonal
  ErrVar=Sigu;
  varLiao=(DF\(Z'))*AF*(kron(ErrVar,eye(T)))*AF*(Z/DF)/(N^2*T^2);
   standard=diag(varLiao).^(1/2); % standard error of betaBai
     for i=1:8  % confidence intervals
          conliao96(i,:)=[betaliao(i)-1.96*standard(i), betaliao(i)+1.96*standard(i)];
         conliao58(i,:)=[betaliao(i)-2.58*standard(i), betaliao(i)+2.58*standard(i)];
      
     end;
     
     
 conbai96 
 conliao96


 
 varliao=diag(varLiao);
 
 
efficiency= varliao./varbai
 betabai
 betaliao
 
 
 
% choosing number of factors

%num= number of factor
for num=1:15
    
 Delta=10;
 F=zeros(T,r);
 L=zeros(N,r);
 G=zeros(8,1);
 for t=1:T
      G=G+X2(:,:,t)'*(y(:,t)-L*F(t,:)');
 end
 betahat=M\G;
 while Delta>10^(-10)
     beta0=betahat;
     for i=1:N
         ZZ(i,:)=y(i,:)-beta0'*X(:,:,i)'; % 1 x T;  Z is N x T
     end;
     [F,L]=factor(ZZ,eye(N),num); % L is N by r  F is T by r
     G=zeros(8,1);
     for t=1:T
         G=G+X2(:,:,t)'*(y(:,t)-L*F(t,:)');
     end;
     betahat=M\G; % inv(matrix)*G
     Delta=norm(betahat-beta0,'fro');
 end;
  
  for t=1:T
     u(:,t)=y(:,t)- X2(:,:,t)*betahat-L*F(t,:)';            %  u(:t) is N by 1    X2(::t) is N x 8,
 end;
  hatsigma(num)=norm(u,'fro')^2/(N*T);
end;
 

 


 for k=1:15
    

   PC2(k)=hatsigma(k)+k*log(N*T)/(N*T)*(N+T)*hatsigma(15);  %  9-13  J=3
  ic(k)=hatsigma(k)+(k*(N+T)-k^2)*log(N*T)/(N*T)*hatsigma(15); %  10-15  J=3
  
 
 
 end;
 

 