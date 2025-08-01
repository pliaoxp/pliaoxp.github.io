 
function [postlava_dense, postlava_sparse, post_lava, lava_dense, lava_sparse, lava_estimate, LAMBDA]=LAVAmatlab(X,Y, K, Lambda1, Lambda2)

% This is a Matlab code for lava and post-lava estimators, where tunings  are chosen by K-fold cross validations.
% The model is: Y= X*b + error, where b can be decomposed into  b = dense + sparse.
% Lava and post-lava are robuste estimations of high dimensional possibly nonsparse models. 
% The method of lava solves the following problem:

%  min_{b2, b1} |Y-X*(b1+b2)|_2^2/n + lambda2*|b2|_2^2+ lambda1*|b1|_1.
% estimator = b2+b1;


% Reference:  Chernozhukov, Hansen and Liao (2015) "A lava attack on the recovery of sums of dense and sparse signals", available on arXiv:1502.03155

% Inputs: 
%    X: n by p, design matrix, where n and p respectively denote the sample    size and the numer of regressors.
%    Y: n by 1, vector of outcomes.
%    K: the "K" fold cross validation.
%    Lambda1: a vector of candidate values to be evaluated for lambda1, in     the cross validation
%    Lambda2: a vector of candidate values to be evaluated for lambda2, in     the cross validation

% e.g.,  set Lambda2 =[0.01, 0.07, 0.2, 0.7,  3, 10, 60, 1000,6000];
%             and   [bb ss]=lasso(X,Y,'NumLambda',50);  Lambda1=ss.Lambda*2;    
%                 or simply Lambda1=[0.01:6/50:6];

%  We recommend use a relatively long vector of Lambda1 (e.g.,50~100 values), but a short vector of Lambda2 (e.g., within 10).
%  Higher dimensions of Lambda2 substantially increase the computational time, because a "for" loop is called for lambda2.

% Outputs: 
%  postlava_dense:   paramter estimate of the dense component using post-lava
%  postlava_sparse:  paramter estimate of the sparse component using post-lava
%  post_lava:        equals postlava_dense+postlava_sparse:  parameter estimate using post-lava
%  lava_dense:       paramter estimate of the dense component using lava
%  lava_sparse:      paramter estimate of the sparse component using lava
%  lava_estimate:    equals lava_dense+lava_sparse:  parameter estimate using lava
%  LAMBDA:  [lambda1lava, lambda2lava,lambda1post, lambda2post]; These are the CV-chosen lambda1 and lambda2 tunings for lava and post-lava.

% Other potential outputs:
% The fitted values for lava and post-lava can be respectively calculated as:  
%   X*lava_estimate  and X*post_lava   

% Example:  


% Lambda2 =[0.01, 0.07, 0.2, 0.7,  3, 10, 60, 1000,6000];
% Lambda1=[0.01:6/50:6];
% K=5;
% n=10; p=5;
% b=zeros(p,1); b(1)=3; b(2:p)=0.1;
% X=randn(n,p); 
% Y=X*b+randn(n,1);
% [postlava_dense, postlava_sparse, post_lava, lava_dense, lava_sparse, lava_estimate, LAMBDA]=lava(X,Y, K, Lambda1, Lambda2);



 % w = warning ('off','all'); % display off
n= length(Y);
p= length(X(1,:));
S=X'*X/n;
[U,M,V]=svd(X); 

    
          
        
     % begin lava and postlava CV process
        for k=1:K
            % training data: size n-n/K
            Y1=[Y(1:(k-1)*n/K);Y((k-1)*n/K+n/K+1:n)];   
            X1=[X(1:(k-1)*n/K,:);X((k-1)*n/K+n/K+1:n,:)];
            S1=X1'*X1/(n-n/K);
            [U1,M1,V1]=svd(X1); 
            % validation data    n/K
            vY=Y((k-1)*n/K+1:(k-1)*n/K+n/K);
            vX=X((k-1)*n/K+1:(k-1)*n/K+n/K,:);

            for l=1:length( Lambda2) % trying different values for lambda2 in the range of Lambda2
          
                 %training for lava
                 H1=M1*inv(M1'*M1+(n-n/K)*Lambda2(l)*eye(p))*M1';
                 Khalf1=U1*sqrtm(eye(n-n/K)-H1)*U1';  % square root of (I- the ridge-projected matrix), which is square root of \tilde K in the paper's notation
                 tY1=Khalf1*Y1; tX1=Khalf1*X1;  % these are transformed data
                 [DELTA fit]=lasso(tX1,tY1,'Lambda',Lambda1); % returns a matrix of vectors of "sparse estimators" for different values of lambda1, chosen from the range Lambda1.
                 BETAlava=(S1+Lambda2(l)*eye(p))\X1'*(Y1*ones(1,length(Lambda1))-X1*DELTA)/(n-n/K); % each column is a vector of "dense estimators".
                  THETA= DELTA+BETAlava;  % p by L   each column is a vector of "LAVA estimators".
                  
                 % validation for lava
                 residual=vX*THETA-vY*ones(1,length(Lambda1));   % each column is a vector of LAVA residuals fitted on the validation data. (again, there are L columns, 
                 %where L=length(Lambda1) represents the number of tried lambda1 values in the range of Lambda1)
                 elava(k,:,l)=sum(residual.^2);  % sum of residual squares % 1 by L; elava(k,b,l)  k: K fold, b: Lambda1's choice, l: Lambda2's choice
                  clear residual fit;
                  
                  
                 % post lava
                 
                 for j=1:length(Lambda1)   % trying different values for lambda1 in the range of Lambda1
                      use=abs(DELTA(:,j))>1e-7;
                       XJ1=X1(:,use);
                       ww=Y1-X1*BETAlava(:,j);
                      % In below,(XJ1'*XJ1)\XJ1'*ww  represents the estimated nonzero elements of the sparse components,using post lava
                      % vX*BETAlava(:,j) represents the fitted dense part on the validation data
                    respost(:,j)=  vX(:,use)*((XJ1'*XJ1)\XJ1'*ww)+ vX*BETAlava(:,j)-vY; % each column is a vector of post-lava residuals fitted on the validation data.
                    clear use XJ1 ww 
                 end;
                 elavapost(k,:,l)=sum(respost.^2);  % sum of residual squares
                   
                 clear residual DELTA THETA  fit  respost;
            end; % l
                  
        end; % k
        
        
        % By now we have evaluated all the lambda1 and lambda2 for all the K possible splitted data sets        
         
            
        % lava fit by choosing the best lambda1 and lambda2, and then refitting the entire data set. 
        for l=1:length( Lambda2)
            CVRlava(l,:)=mean( elava(:,:,l)); %CVRlava(l,g): l: Lambda2(l)   g:  Lambda1(g) 
        end;
         [a,b]=min(CVRlava);
         [c,d]=min(a);  
         lambda2=Lambda2(b(d)); lambda1=  Lambda1(d); % optimal choice of lambda1 and lambda2 for lava
         lambda1lava=lambda1; lambda2lava=lambda2;
          P=X/(S+lambda2*eye(p))*X'/n;     Kmatrix=eye(n)-P;
          H=M*inv(M'*M+n*lambda2*eye(p))*M';
          Khalf=U*sqrtm(eye(n)-H)*U'; % square root of Kmatrix
           tY=Khalf*Y; tX=Khalf*X;  % transfored data
           [deltahat,stats]=lasso(tX,tY,'Lambda',lambda1); % deltahat is the final estimator of the sparse component
           dense=(S+lambda2*eye(p))\X'*(Y-X*deltahat)/n;  % final estimator of the dense component
           lavatheta=deltahat+dense; % final estimator of lava 
           % Prelava= P*Y+Kmatrix*X* deltahat;   % this is just an equivalent expression for X*lavatheta. One can also check if Prelava is almost the same as X*lavatheta, making sure the code is correct
             
               % I am used to clearing intermediate variables in the middle of calculations. This is likely not necessary, but may avoid potential errors due to potential dimension changes of cerntain variables.
               
               clear CVRlava a b c d lambda2 lambda1 P H Khalf tY tX Kmatrix   stats Prelava
               
     
         % Post-lava fit by choosing the best lambda1 and lambda2, and then refitting the entire data set. 
         for l=1:length( Lambda2)
                     CVRpostlava(l,:)=mean(  elavapost(:,:,l)); %CVRpostlava: l: Lambda2(l)   g:  Lambda1(g) 
         end;
         [a,b]=min(CVRpostlava);
         [c,d]=min(a);  
         lambda2=Lambda2(b(d)); lambda1=  Lambda1(d); % optimal choice of lambda1 and lambda2 for post-lava
         lambda1post=lambda1; lambda2post=lambda2;
          P=X/(S+lambda2*eye(p))*X'/n;   %  Kmatrix=eye(n)-P;
          H=M*inv(M'*M+n*lambda2*eye(p))*M';
          Khalf=U*sqrtm(eye(n)-H)*U';
          tY=Khalf*Y; tX=Khalf*X;   % transfored data
          [deltahat,stats]=lasso(tX,tY,'Lambda',lambda1);
          post_dense=(S+lambda2*eye(p))\X'*(Y-X*deltahat)/n;  %  post-lava estimator of the dense component  
          use=abs(deltahat)>1e-7;
          XJ=X(:,use);
          post_sparse=zeros(p,1); 
          post_sparse(use)=pinv(XJ'*XJ)*XJ'*(Y-X*post_dense); %  post-lava estimator of the sparse component   
          post_lavatheta=  post_dense+post_sparse;  % final estimator of post-lava 
         % Prelava= P*Y+Kmatrix*X* deltahat;  
         % PJ=XJ*pinv(XJ'*XJ)*XJ'; 
         % pre_post=Prelava+PJ*(Y-Prelava);   % this is just an equivalent expression for X*post_lavatheta. One can also check if  pre_post is almost the same as X*post_lavatheta, making sure the code is correct
            
         clear CVRpostlava use elavapost XJ
         clear a b c d lambda2 lambda1 P Kmatrix H Khalf tY tX   stats Prelava  CVRlava
         clear  Preridge  thetaridge   lambdaridge  CVRridge elava THETA DELTA fit tY1 tX1 Khalf1 H1 eridge ridge vX vY
         clear U1 M1 V1 S1 X1 Y1  2 LAM Lambdar center Prelasso  thetalasso bb ss  
         clear XJ PJ pre-post  epostlava Preenet1 Preenet2
                  
               
         % outputs:

lava_dense=dense;
lava_sparse=deltahat;
lava_estimate=lavatheta;
postlava_dense=post_dense;
postlava_sparse=post_sparse;
post_lava=post_lavatheta;
LAMBDA= [lambda1lava, lambda2lava,lambda1post, lambda2post]; 
