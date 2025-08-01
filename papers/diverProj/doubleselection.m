function   [ betahat , se ]=   doubleselection(R, X,Y, G, weight, lambday , lambdag)

% R :  num factors in X.   R= 0 if do NOT estimate factors. 

% X is predictor, with factor struture 
% G is treatment variable, whose effect is of interest 
% Y is outcome 
%   lambday ,   lambdag tuning for lasso 
% weight: N by bigR, weights for DP

[N,T] = size(X); 

  

    if R==0 
        Uhat = X ;
        newY =Y; 
        newG= G; 
    else % R>0
         % step 1 estimate factors in X 
                 
        W=weight(:,1:R);
        F_est= X'*W/N ; % T by R
        S_Fest= F_est'*F_est/T ; 
        L_est= 1/T*X*F_est/ S_Fest ; 
        Uhat= X- L_est*F_est'; % N by T
        
             
        % step 2 estimate alpha for Factors
        alphaY= inv(S_Fest)*F_est'*Y/T; 
        alphaG= inv(S_Fest)*F_est'*G/T; 
        
    
      % step 3 lasso and post lasso
        newY= Y- F_est*alphaY;
        newG= G- F_est*alphaG;

    end % if R
       
   %% step  4 double selection 

                [gammahat,~]=lasso(Uhat',newY,'Lambda',lambday);
                [thetahat,~]=lasso(Uhat',newG,'Lambda',lambdag);
                select= (abs(gammahat)>1e-8)+ (abs(thetahat)>1e-8)>0;
                newU= Uhat( select,:); 
                MM=(newU*newU')\newU;
                newgamma=MM *newY;
                newtheta=MM *newG;
                
                %%  step 5 residual regression
                resy=  newY -  newU'*newgamma;
                resg=  newG -  newU'*newtheta;
 
                betahat =(resg'*resg)\resg'*resy;

                %% z stats
                sigmag= resg'*resg/T;
                etahat=  resy -  betahat*resg;
                sigmaeta=mean((etahat.^2 ).*  ( resg.^2));
               se= sqrt( sigmaeta)/sigmag /sqrt(T); % std error
 
                