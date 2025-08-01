
function [SigmaY,SigmaU]=POET(Y,K,C, thres)
 
 
%%%%% This function is for POET, proposed by Fan, Liao and Mincheva (2012)
%%%%%  "Large Covariance Estimation by Thresholding Principal
%%%%%    Orthogonal Complements", JRSSB discussion paper.

%%%%% We are grateful to Michael Wolf, who has implemented our code on 30~500 stocks  
%%%%% based on T=252 daily returns. He found that 

%%%%%    K=5 factors 
%%%%%    C=1.0 for the thresholding constant, and 
%%%%%    for the soft thresholding

%%%%% works well on average. These parameters are to be explained below.
 
%%% Model:  Y_t=Bf_t+u_t, where B, f_t and u_t represent factor loading
%%%%%        matrix, common factors and idiosyncratic error respectively.
%%%%%        Only Y_t is observable. t=1,...,n. Dimension of Y_t is p. The
%%%%%        goal is to estimate covariance matrices of Y_t and u_t.

%%% Note: (1) POET is optimization-free, so no initial value, tolerant, or
%%%%%         maximum iterations need to be specified as inputs.

%%%%%     (2) If no factor structure is assumed, i.e., no common factor
%%%%%         exists and var(Y_t) itself is sparse, set K=0.


%%% Inputs:
%%%%%  Y:    an p by n matrix of raw data, where p is the dimensionality, n
%%%%%        is the sample size. It is recommended that Y is de-meaned,
%%%%%        i.e., each row has zero mean.

%%%%%  K:    number of factors. K is pre-determined by the users. Suggestions on choosing K:
%%%%%     (1) A simple way of determining K is to count the number of
%%%%%        very spiked (much larger than others) eigenvalues of the
%%%%%        p by p sample covariance matrix of Y. 
%%%%%     (2) A formal data-driven way of determining K is described in
%%%%%        Bai and Ng (2002):
%%%%%       "Determining the number of factors in approximate factor
%%%%%       models", Econometrica, 70, 191-221. This procedure requires a
%%%%%       one-dimensional optimization.
%%%%%     (3) POET is very robust to over-estimating K. But under-estimating K
%%%%%         can result to VERY BAD performance. Therefore we strongly recommend
%%%%%         choosing a relatively large K (normally less than 8) to avoid
%%%%%         missing any important common factor.
%%%%%     (4) K=0 corresponds to threshoding the sample covariance directly

%%%%%  C:   the positive constant for thresholding, user-specified. Our experience shows that
%%%%%       C= 0.5~1 performs well for soft thresholding.

%%%%%  thres: the option of thresholding. Users can choose from three
%%%%%         thresholding methods:
%%%%%        'soft': soft thresholding
%%%%%        'hard': hard thresholding
%%%%%        'scad': scad thresholding
  



%%% Outputs:
%%%%%  SigmaY:    estimated p by p covariance matrix of y_t
%%%%%  SigmaU:    estimated p by p covariance matrix of u_t

%%% Toy example:  
%%%%%             p=100; n=100;
%%%%%             Y=randn(p,n);
%%%%%             [Sy,Su]=POET(Y,3,0.5, 'soft');




[p,n]=size(Y);
Y= Y-mean(Y')'*ones(1,n) ; % Y is de-meaned
if K>0
    [V,Dd]=eig(Y'*Y);
    index= sort((n-K+1:n),'descend');
    F=V(:,index)*sqrt(n);
    LamPCA=Y*F/n; 
    uhat=Y-LamPCA*F';   % p x n
    Lowrank=LamPCA*LamPCA';
    rate=1/sqrt(p)+sqrt((log(p))/n);

else uhat=Y; % Sigma_y itself is sparse
    rate=sqrt((log(p))/n);
    Lowrank=zeros(p,p);
end;


lambda=rate*C*ones(p,p);

SuPCA=uhat*uhat'/n;
SuDiag=diag(diag(SuPCA));
R=inv(SuDiag^(1/2))*SuPCA*inv(SuDiag^(1/2));       
switch thres
    case 'soft'
        M=wthresh(R,'s',lambda);
    case 'hard'
        M=wthresh(R,'h',lambda);
    case 'scad'
        M1=(abs(R)<2*lambda).*sign(R).*(abs(R)-lambda).*(abs(R)>lambda);
        M2=(abs(R)<3.7*lambda).*(abs(R)>=2*lambda).*((3.7-1)*R-3.7*sign(R).*lambda)/(3.7-2);
        M3=(abs(R)>=3.7*lambda).*R;
        M=M1+M2+M3;
       
end;
        
  Rthresh=M-diag(diag(M))+eye(p);
  SigmaU=SuDiag^(1/2)*Rthresh*SuDiag^(1/2);
  SigmaY=SigmaU+Lowrank;
    
  
  
