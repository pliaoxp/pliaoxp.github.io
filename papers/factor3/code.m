%Background: This code implements the joint-estimation of the penalized ML for the
%             factors and loadings, proposed by Bai and Liao (2012): 
%            "Efficient Estimation of Approximate Factor Models via Regularized Maximum Likelihood"
%            We use the EM+MajorizeMinimize method proposed by Bai and Li (2012) and
%             Bien and Tibshirani (2011).

% Initial value: Our algorithm uses the Strict Factor Model (SFM) estimator
%                of Bai and Li (2012) as an initial value. The latter uses PCA of Stock
%                and Watson (1998) as an initial value.

% Function calls: Please save everything in the same folder. The code loads
%                 them

% This file gives the code for the simulation in our main paper.

%%%%%%%%%%%%% Model %%%%%%%%%%%


s = RandStream('mt19937ar','seed',sum(100*clock));
RandStream.setDefaultStream(s);

 
r=2;   % num of factors 
T=100;
N=150;
for i=1:N
    a(i)=randn*0.7;
    b(i)=randn*0.7;
    c(i)=randn*0.7;
end;
Strue=zeros(N,N);
for i=3:N-1
    Strue(i+1,i+1)=1+a(i)^2+b(i-1)^2+c(i-2)^2;
    Strue(i+1,i)=a(i)+b(i-1)*a(i-1)+c(i-2)*b(i-2);
    Strue(i,i+1)=Strue(i+1,i);
    Strue(i+1,i-1)=b(i-1)+c(i-2)*a(i-2);
    Strue(i-1,i+1)=Strue(i+1,i-1);
    Strue(i+1, i-2)=c(i-2);
    Strue(i-2,i+1)=Strue(i+1,i-2);
end;
Strue(1,1)=1; Strue(2,2)=1+a(1)^2; Strue(3,3)=1+a(2)^2+b(1)^2;
Strue(1,2)=a(1); Strue(2,1)=Strue(1,2);
Strue(1,3)=b(1); Strue(3,1)=Strue(1,3);
Strue(2,3)=a(2)+a(1)*b(1); Strue(3,2)=Strue(2,3);
Strue;

for i=1:N
    for j=1:r
        Ltrue(i,j)=unifrnd(0,1);% Ltrue is N x r
    end;
end;

for or=1:500  % number of replications
    e=randn(N,T);  % e is N x T   %  u= N x T
    for i=3:N-1
        u(i+1,:)= e(i+1,:)+a(i)*e(i,:)+b(i-1)*e(i-1,:)+c(i-2)*e(i-2,:);
    end;
    u(3,:)=e(3,:)+a(2)*e(2,:)+b(1)*e(1,:);
    u(2,:)=e(2,:)+a(1)*e(1,:);
    u(1,:)=e(1,:);

    Ftrue(:,:,or)=randn(T,r);  % factor   % T x r
    YY(:,:,or)=Ltrue*Ftrue(:,:,or)'+u;   % N x T % Y=YY(:,:,or);
    clear u e;
end;
 
limit=size(YY);
T=limit(2);
N=limit(1);
Lambdatrue=Ltrue;

%%%%%%%% Initial values

 
for or=1:limit(3)
    % Calcualte  PCA

    Y=YY(:,:,or); % Y is N by T
    [V,D]=eig(Y'*Y);
    [W,I]=sort(diag(D));
    for i=1:r
        F(:,i)=sqrt(T)*V(:,I(T-i+1));   % F is T x r
    end;
    LamPCA=Y*F/T;  % N x r
    uhat=Y-LamPCA*F';   % N x T
    SuPCA=uhat*uhat'/T;
    LPCA(:,:,or)=LamPCA;
    SigPCA(:,:,or)=SuPCA;
    clear   V D W I   uhat;
    Ybar=mean(Y')'*ones(1,T);
    
     %%%%%%%%%%% Canonical correlation of PCA estimators
    [A2,B2,coPCA] = canoncorr(Lambdatrue,LPCA(:,:,or)); 
    [cc2,cc2,ccPCA] = canoncorr(F,Ftrue(:,:,or)); 
     absPCA(or)=min(abs(coPCA));  % PCA loadings
     acc(or)=min(abs(ccPCA));  % PCA factors
 
     
     %%%%%%%%%%%% Calculate SFM (diagonal Max Likelihood of Bai and Li 2012) as initial value 
     %%%%%%%%%% SFM estimate uses PCA as initial value
    Ybar=mean(Y')'*ones(1,T);
    Sy=Y*Y'/T;
    kk=1;
    Lambda0=ones(N,r)*10;
    Lambda=LPCA(:,:,or);
    Su=SigPCA(:,:,or);
    Sigma1=diag(diag(Su));
    Sigmaold=eye(N)*100;

  while likelihoodlambda(Sy,Sigmaold,Lambda0)-likelihoodlambda(Sy,Sigma1,Lambda)>10^(-7)&&kk<4000
    Sigmaold=Sigma1;  
    Lambda0=Lambda;  
    A=inv(Lambda0*Lambda0'+Sigma1);
    C=Sy*A*Lambda0;
    Eff=eye(r)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C*inv(Eff);  % N x r
    M=Sy-Lambda*Lambda0'*A*Sy;  % Sy-C'
    Sigma1=diag(diag(M));
    kk=kk+1 ;
  end;
    Factor= (inv(Lambda'*inv(Sigma1)*Lambda)*Lambda'*inv(Sigma1)*(Y-Ybar))';  % T by r
    
    %%%%%%%%% Canonical correlations of SFM estimators
    [cD,cSA,ccML] = canoncorr(Factor,Ftrue(:,:,or)); 
    [A1,B1,coML1] = canoncorr(Lambdatrue,Lambda); 
    ccMldd(or)=min(abs(ccML)); % SFM factors
    absMLd(or)=min(abs(coML1));  % SFM loadings
    LML(:,:,or)=Lambda; 
    SigML(:,:,or)=Sigma1;
    FactorML(:,:,or)=Factor;

  clear Sigma1 Sigma0new Sigmanew A Sigma Sigma0 Su Eff Su0 C Lambda0 LamPCA Lambda  uhat  F  Y SuPCA Sutrue;
  clear A1 B1  ccML   coML1     ;
end;

%%%%%%%%% Up to now, we obtain initial values LML and SigML.


%%%%%%%%%%% EM + Majorize Minimize algorithm ( joint estimation in our main paper)

 lambda=0.08;  % tuning parameter
 gamma=5;   % tuning parameter
 t=0.1;  % tuning parameter
 
 
for or=1:limit(3)
    Y=YY(:,:,or);
     Ybar=mean(Y')'*ones(1,T);
    Sy=Y*Y'/T;
    Lambda=LML(:,:,or);
    Su=SigML(:,:,or);
    Sigmaold=Su;
    Lambda0=Lambda;
    A=inv(Lambda0*Lambda0'+Sigmaold);
    C=Sy*A*Lambda0;
    Eff=eye(r)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C*inv(Eff);
    Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda'; 
    KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
    P=Pmatrix(Su,gamma);
    B=lambda*t*P;
    Sigma1=soft(KML,B);
    kk=1;
  while  likelihoodTrue(Sy,P,Sigmaold,Lambda0,lambda)- likelihoodTrue(Sy,P,Sigma1,Lambda,lambda)>10^(-7)&kk<5000
    Sigmaold=Sigma1;
    Lambda0=Lambda;  % old
    A=inv(Lambda0*Lambda0'+Sigma1);
    C=Sy*A*Lambda0;
    Eff=eye(r)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C*inv(Eff);
    Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda';
    KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
    P=Pmatrix(Su,gamma);
    B=lambda*t*P;
    Sigma1=soft(KML,B);
    kk=kk+1;
  end;
  Lambda=Lambda0;
  Sigma1=Sigmaold;
  Factor= (inv(Lambda'*inv(Sigma1)*Lambda)*Lambda'*inv(Sigma1)*(Y-Ybar))';  % T by r
  % Canonical correlations
   [A1,B1,coML1] = canoncorr(Lambdatrue,Lambda); 
  [cdD,cSdA,ccPen] = canoncorr(Factor,Ftrue(:,:,or)); 
  absML(or)=min(abs(coML1));  % canonical correlation of loadings
  ccPe1(or)=min(abs(ccPen));  % canonical correaltion of factors;
  clear   Sigma0new Sigmanew A Sigma Sigma0 Su Eff Su0 C Lambda0 LamPCA Lambda  uhat  F Y SuPCA Sutrue;
  clear A1 B1 A coML1 coPen cdD cSdA u;

end;


% PCA
mean(absPCA)  % lambda
mean(acc)   % F 

% SFM based on diagonal ML, Bai and Li 2012
mean(absMLd)
mean(ccMldd)   %F

% Joint estimation based on penalized ML
mean(absML)  % lambda
mean(ccPe1)  % F


 

