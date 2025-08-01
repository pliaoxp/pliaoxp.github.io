function  forecast=forecastmethod(  R, Y, X,Weight )

% R: num working factors 
% Y: N by T "big data" to estimate factors
% X:  outcome to forecast, T  by 1 
% Weight: N by bigR  weight , bigR>= R 

[N,T]=  size(Y); 

%% insample data: 

lag = X(1:T-1); 
outcome= X(2:T);
 
%% out of sample data 
lagout= X(T) ; % as out of sample data point


 
        
        % estimate factors
        W=    Weight(:,1:R);
        F_est= Y'*W/N ; % T by R
        f=[F_est(1:T-1,:),lag ,ones(T-1,1) ]  ; % T-1 by R+2
        deltahat=(f'*f)\f'*outcome;
        forecast=    [F_est(T,:), lagout,1]* deltahat ;
        
     