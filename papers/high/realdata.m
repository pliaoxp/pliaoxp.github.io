% The model: 
 %%%%%  y= X'beta+error
 % X=(X_S, X_N)
 % Assuming E(error|X_S)=0, where X_S is the low dimensional vector of
 % important regressors. So the unimportant regressors can be endogenous
 % No additional instrumental variables are needed.
 % The goal is to identify X_S
 
 
%(1) you need to imput y and X first. In your input, y is n by 1, X is p by n
% n: sample size
% p: dimension, number of regressors


%(2) The initial value for beta is given in line 51: betaold. If n>p, it is strongly
% recommended to use OLS as the initial value to speed up. In that case,
% change betaold on line 24 to:
%%%% betaold=(X*X')\X*y;

%(3) This code uses SCAD penalty for FGMM, with least square+SCAD as initial
% s2=0.1 should not be changed
% lambda1=0.01 is used for an initial, this needs not be changed
% lambda=0.01 is the default tuning parameter for FGMM. A cross validation
% procedure can be applied to choose this lambda.

%(4)q2(i,j)=0 is the default value. I would not recommend change it. But an alternative is 
% q2(i,j)=X(i,j)^2*0.001;  
% One can run the program twice: with q2=0 and q2=X^2*0.001 respectively,
% and compare the results obtained. Ideally, the obtained sparse solutions
% should be close to each other. If not, then take the one with
% q2=X^2*0.001

%(5) Put everything in the same folder, and run this file.

%(6) If X_S is also possibly endogenous, you need to have an additional
% instrumentle variale Z so that E(error |Z)=0.  Then replace q1 and q2
% below with transformations of Z, such as polynomial, log, Fourier, etc.




y=      % n by 1
X=      % p by n
 
 
s2=.1;
lambda1=0.1;  % initial value, cannot be large, but if it is too small, it takes forever to run
lambda=0.1;          % FGMM, can be chosen using cross validation



betaold=(X*X'+0.1*eye(p))\X*y; % initial value for beta. Change to betaold=zeros(p,1)  or betaold=(X*X'+0.5*eye(p))\X'*y if n<=p



for i=1:p
    for j=1:n
        q1(i,j)=X(i,j); 
        %q2(i,j)=X(i,j)^2*0.001;
        q2(i,j)=0;
         
    end;
end;

q=[q1;q2];

inv(X)*3;

k1=0;

m=10^15;
mnew=10^20;
 
while abs(log(m)-log(mnew))>10^(-7) && k1<3000
    mnew=m;
   
      for j=1:p
          bj=findPLS(j,betaold,y,X,lambda1);
          betanew=betaold;
          betanew(j)=bj;
          gk=PLS(betanew,y,X,lambda1);
          if gk<PLS(betaold,y,X,lambda1)
              betaold(j)=bj;
              m=gk;
          end;
      end;
      
    k1=k1+1;
end;

 
    initial=betaold;
    k=0;

    m=10^15;
    mnew=10^20;
    while abs(m-mnew)>10^(-7) && k<3000
        
        mnew=m;
        for j=1:p
                bj=findGMM(j,betaold,q1,q2,y,X,lambda,s2) ;
                betanew=betaold; betanew(j)=bj;
                m1=objGMM(X,y,betanew,q1,q2,s2,lambda);  
                if m1<m
                    betaold(j)=bj;
                    m=m1;
                end;
        end;
        k=k+1;
       
    end;



    bOLS=initial; % OLS result 
    bGMM=betaold; % FGMM result
    
    pgmm=0; 
    for i=1:p
         
        if abs(bGMM(i))>10^(-5)
             
                pgmm=pgmm+1; % number of selected variables
    
           
        end;
    end;



 