function [thetaLAVA,thetaPOST, compoLAVA,  compoPOST,DFLAVA,lambdaLAVA,lambdaPOST]=CVlava(X,Y,lambda1range,lambda2range,K, post)
%%%% This function calculates the lava estimation, where tuning parameters are chosen by K-fold cross validation. It uses a built-in
%%%% function "lava", which is self-contained, and could be used independently.
%%%% In addition, this function also calculates the post-lava estimator, where the tuning paramters are also chosen by K-fold cross validation.

% need "bioinformatics" toolbox to use `` crossvalind"

%%%% The model: Y=X*theta+ error

%%%% LAVA estimator is defined as:

%%%%  theta=beta+delta, where
%%%%  min_{beta, delta} 1/n*|Y-X*(beta+delta)|^2 +lambda2*|beta|_2^2 +lambda1*|delta|_1
 
%%%% Interpretation: (1) The LAVA method is proposed by Chernozhukov, Hansen and Liao (2015). 
%%%%                 (2) LAVA strictly dominates Lasso and Ridge. When the true signal is sparse, Lava performs as good as lasso; when the true signal is dense, 
%%%%                     Lava performs as good as ridge. When the true signal is neither sparse nor dense, 
%%%%                     Lava is strictly better than lasso, ridge, and elastic net.
%%%%                 (3) In particular, in the self-contained "lava" function, by setting lambda2= a large value (e.g., 1e+10),  it outputs the lasso estimator. 
%%%%                     By setting lambda1= a large value (e.g., 1e+10),  it outputs the ridge estimator.
%%%%                 (4) The beta and delta respectively represent the "dense" and "sparse" components of theta.

        %%%% Inputs: lambda2range:   A range of lambda2 values, used for the grid-search in the cross validation process. 
        %%%%                         An example: lambda2=[1e-4, 0.008, 0.1, 2, 30, 160, 530, 1000, 5000,1e+5];
        %%%%         lambda1range:  A range of lambda1 values, used for the grid-search in the cross validation process. 
        %%%%                         An example: [b, s]=lasso(X,Y); lambda1=s.Lambda;
        %%%%         X:     n by p
        %%%%         Y:     n by 1      
        %%%%         K:     scalar, for the "K"-fold cross validation.
        %%%%         post:  If post-LAVA is also calculated using cross validation, set post=1; otherwise set post to any  other number. Then the post-LAVA uses the same tunings of LAVA.
        %%%%             Note: setting post=1 potentially can improve the accuracy of the post-LAVA estimation, but takes longer time.
        %%%%       
        %%%% Outputs:  
        %%%%          thetaLAVA    =betaLAVA+deltaLAVA:  lava estimator     
        %%%%          thetaPOST    =betaPOST+deltaPOST:   post-lava estimator
        %%%%          compoLAVA=[betaLAVA,deltaLAVA],      estimated dense  components and estimated sparse components using LAVA
        %%%%          compoPOST=[betaPOST,deltaPOST],      estimated dense components and estimated sparse components using POST-LAVA, where the biased introduced by L_1 shrinkage is removed.
        %%%%          DFLAVA:       The degree of freemdom of LAVA.
        %%%%        lambdaLAVA=[lambda2, lambda1], CV selected tunings for LAVA
        %%%%        lambdaPOST=[lambda2, lambda1], CV selected tunings for POST- LAVA; 
        %%%%                    If input post is not 1,    lambaPOST=lambdaLAVA.

          [n,p]=size(X);
          Indices = crossvalind('Kfold', n, K);
 
    if post==1  % post-lava is also calculated using K-fold CV
         for l=1:length( lambda2range)
             for k=1:K
                    % training data:  
                    Y1= Y(Indices==k); 
                    X1= X(Indices==k,:); 
                      % validation data     
                    vY=Y(Indices~=k);
                    vX=X(Indices~=k,:);  
                    
                 [theta,thetapost, beta,delta1, delta2,DF1]=lava(X1,Y1,lambda2range(l),lambda1range, 1);
                 residual=vX*theta-vY;  
                 respost=vX*thetapost-vY;
                 elava(k,:,l)=sum(residual.^2,1);
                 elavapost(k,:,l)=sum(respost.^2,1);
             clear Y1 X1   vX vY residual   respost
             end; % K
             CVRlava(l,:)=mean( elava(:,:,l),1); %CVRlava(l,g): l: Lambda2(l)   g:  Lambda1(g)
             CVRpostlava(l,:)=mean(  elavapost(:,:,l),1); 
         end; % lambda2range
         % lava tuning
         [a,b]=min(CVRlava);
         [c,d]=min(a);  %index= [  b(d), d]
         lambda2lava=lambda2range(b(d)); lambda1lava=  lambda1range(d);
           % post lava tuning
           [a1,b1]=min(CVRpostlava);
         [c1,d1]=min(a1); 
         lambda2post=lambda2range(b1(d1)); lambda1post=  lambda1range(d1);
         [thetaLAVA,thetadd, betaLAVA,deltaLAVA, deltadd,DFLAVA]=lava(X,Y,lambda2lava,lambda1lava, 0);
         [thetfrgA,thetaPOST, betaPOST,delfgdVA, deltaPOST,DgfdAVA]=lava(X,Y,lambda2post, lambda1post, 1);
          lambdaLAVA=[lambda2lava,lambda1lava];  
          lambdaPOST=[lambda2post,lambda1post];
          compoLAVA=[betaLAVA,deltaLAVA];
          compoPOST=[betaPOST,deltaPOST];

    else  % post-lava is not calculated using K-fold CV
        
         for l=1:length( lambda2range)
             for k=1:K
                    % training data:  
                    Y1= Y(Indices==k); 
                    X1= X(Indices==k,:); 
                      % validation data     
                    vY=Y(Indices~=k);
                    vX=X(Indices~=k,:);  
                    
                 [theta,thetapost, beta,delta1, delta2,DF]=lava(X1,Y1,lambda2(l),lambda1range, 0);
                 residual=vX*theta-vY;  
                 elava(k,:,l)=sum(residual.^2,1);
             clear Y1 X1   vX vY residual   respost
             end; % K
             CVRlava(l,:)=mean( elava(:,:,l),1); %CVRlava(l,g): l: Lambda2(l)   g:  Lambda1(g)
         end; % lambda2range
         % lava tuning
         [a,b]=min(CVRlava);
         [c,d]=min(a);  %index= [  b(d), d]
         lambda2lava=lambda2range(b(d)); lambda1lava=  lambda1range(d);
         [thetaLAVA,thetaPOST, betaLAVA,deltaLAVA, deltaPOST,DFLAVA]=lava(X,Y,lambda2lava,lambda1lava, 1);
          lambdaLAVA=[lambda2lava,lambda1lava];
          lambdaPOST=lambdaLAVA;
           compoLAVA=[betaLAVA,deltaLAVA];
           compoPOST=compoLAVA;
    end; % post
    
    
    
    
end
             
 
             

        
        