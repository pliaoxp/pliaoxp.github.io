

%% sample code to run this function:

%{

N = 100;
T= N-1; 
d =2 ; % number of regressor X
Ex= randn(N,T,d) ; 
X= Ex +1; 
Y= 1  +  sum(X,3)+  randn(N,T); 
commonmethod = 1; 
stdu=0; 
group= [1:30;31:59 , 0 ]' ;
[thetahat, sehat, stergroup,groupeffecthat, fixeffecthat,pvalue] = main_fun(Y,X, commonmethod, stdu,group) ;

%}

 
function [thetahat, sehat, stergroup,groupeffecthat, fixeffecthat,pvalue] = main_fun(Y,X, commonmethod, stdu,group) 


%% input 

%  Y: N by T
%   X: N by T by d balanced panel
%  commonmethod: the method used to decompose X

        % X = common + E, where E is error components in X; common is the "common" part in X,
        %  like mean, or PC or common trend 
        
        % commonmethod =1 : TS average: common = average sum_t X_it / T
        % commonmethod =2 : CS average: common = average sum_i X_it / N
        % commonmethod =3 : common = two-way-mean + two principal components of X for each regressor in X 


%  stdu: a positive scalar in the tuning parameter, default choice  sets stdu =0,  if so, then the code 
%         will compute it for you (as the averaged estimated std of the residual)
%         if the code does not give you good result or produce errors, try setting your own stdu, 
%        by a small number, say stdu= 0.02. If the code continues failing, try lowering stdu. 


% group: GG by M:  GG is the number of individuals in each group (group size), M is the number of groups 
      %  each column of "group" is indices of individuals for each group. If the group size is
          %         not the same, GG is the max group size, 
          % and fill in "0" to the LAST positions of each column 
          % For instance, say two groups: Group 1: individuals {1,3}; and group 2: individuals {2,5,6}. 
          % Then group= [  1,2;3,5;0,6  ] and GG=3.
          % set group =0 if not estimate any group

%% Output 

% thetahat: N by d by T, the estimated effect, at all periods, individuals, and regressors
% sehat: N by d by T:  standard error for each estimated effect
% stergroup:  num of groups by d by T: group standard error,
%           if input group =0 then stergroup=0
%  groupeffecthat: num of groups by d by T: group average effect: simple averages of theta within each group. 
 
% fixeffecthat: N by T: interactive fixed effect 
% pvalue: num of group by 1:  p value for testing group homogeneity 
            % To test within-group effect homogeneity, over all periods, for a particular regressor X^*


          %% Notes: 

% (1) the code is designed so that Y and X are both N by T, where N is  individuals and T is time periods. 
% %        This assumes residuals are independence over time . 
% (2) But you can also try input Y as T by N, and X as T by N by d, then the code will also run , 
%       which will assume residuals are independent over individuals. 
%     The outputs "theta" and "se" and "fixeffect" still work. 
%     The output theta and se should be understood as T by d by N then. Try both   to see whichever works better. 
% 
% (3) If you decide to use the input as T by N and T by N by d matrices, then "group" should be understood as by time
%      (e.g., before and after  covid). If you however still want to group by individuals, then set group =0 ,
%       and you should manually take average of etimated theta over   indivduals of your group. But computing 
%      the group se "stergroup" is more challenging, the code does not do it for you unfortunately. 
%     
% (4) this code fixes Keffect =1, the number of factors in each slope (assumed the same). 
% %    You can try other Keffect by changing in the code, but do not use  too big

% (5) To test within-group effect homogeneity for a particular regressor X^* , you should put X^* as the
%            first X, meaning X(:,:,1) = X^*

%% num factors in slope 


 Keffect =1 ; % num factors in each slope 
numbPCX = 2;  % num factors in each X, if commonmethod ==3: use PCA to estimate the structure of X 
% the code estimates num factors in the interactive fixed effect for you 

%% load data 

% after loading data , you should have 

% Y: N by T
% X: N by T by d, where d is number of regressors 

d= size(X,3); 
T= size(X,2) ;
N= size(X,1) ; 



%% create common  in X, so that 
 

rowMean = mean(X,2);          % N x 1 x d
colMean = mean(X,1);          % 1 x T x d
grandMean = mean(mean(X,1),2); % 1 x 1 x d
Xdemean = X - colMean - rowMean + grandMean;  


switch commonmethod 
    case 1 
        Common= repmat(mean(X,2), [1, T, 1]);
    case 2 
         Common= repmat(mean(X,1), [N, 1, 1]);
    case 3 % PCA
        
        Common= nan(N,T,d);
        for r = 1:d
            newXX= X(:,:,r) ; 
            newN=length(newXX(:,1));
            Z=Xdemean(:,:,r) ;
             Sam_cov= Z*Z'/T; 
           [tempV,tempD]         =      eig(Sam_cov/newN);   
           [tempeig,tempidx]     =      sort(real(diag(tempD)),'descend');
           eigvec                =      tempV(:,tempidx); 
           u1                    =      eigvec(:,1:numbPCX);
           l = u1*sqrt(newN);  % newN by Kx
           w =   Z'*l/newN; % T by Kx
           factorstructure= l*w';
           newE =  Z- factorstructure ;
           Common(:,:,r) = newXX -newE ;
          
        end 

end % switch 


 %% decide tuning parameters 

Ydemean = Y  - mean(Y,1)  - mean(Y,2)     + mean(Y(:));

if stdu ==0
    XX = reshape(Xdemean , N*T, d);     % (NT) x d
    YY = reshape(Ydemean , N*T, 1);     % (NT) x 1
    beta_hat = XX \ YY;
    Yhat = reshape(XX * beta_hat , N, T);
    noise = Ydemean - Yhat;
    stdu = std(noise(:)) ;
end 
 
 
% tuning0 produced in this code will be for estimating the intercept fixed  effect 
% tuning will be for estimating the slope fixed effect. It has two elements, each for half of the time periods used in sample splitting
tuning=zeros(d,1);
for r=1:d
    tuning(r)= norm(X(:,:,r).*randn(N,T)*stdu,2)*1.1;
end % r 
tuning0= norm(randn(N,T)*stdu,2)*1.1;
  

%% estimation

 
 sehat= nan(N,d,T); 
 thetahat=nan(N,d,T); 
 stergroup= nan(size(group,2),d,T); 
 fixeffecthat= nan(N,T) ;
 betahat = nan(N,Keffect,T) ;  % for testing within group homo
  var_betahat= nan(N,Keffect,T) ;   % for testing within group homo
 Corrmatrixhat= nan(Keffect,Keffect,T) ; % for testing within group homo


      %{
%% these are needed for future updates of the code: to add test of group homogeneity
 
 
pvaluehat= nan ; 
      %}

 

  parfor time = 1:T

     [thetahat(:,:,time),  sehat(:,:,time), stergroup(:,:,time), ...
         fixeffecthat(:,time),betahat(:,:,time),  var_betahat(:,:,time), Corrmatrixhat(:,:,time)]...
         = factorslopeestimate(Y,X, Common , Keffect, time,   tuning,  tuning0, group) ;
  
  
end % time 
 


%% group average effect 


if norm(group)>0
    M = size(group, 2) ; 
     groupeffecthat= nan(d,T,M) ; 
    for r= 1:M
        indiv = group(:,r) ; 
        slopeffect = thetahat(indiv(indiv~=0), :, :);
       groupeffecthat(:,:,r) = reshape(mean(slopeffect,1), d, T);
    end 
   
end % if norm(group)>0


%% within group homogeneity  test 


loading =  mean(betahat,3) ; % N by K(1)
var_loading =  mean( var_betahat,3) ; % N by 2
pvalue= nan(M,1) ;

if norm(group)>0
     M = size(group, 2) ; 
     Corrmatrix_final = mean(Corrmatrixhat,3)  ;  
     sqcov=  Corrmatrix_final^(1/2) ;
        
    for r =1:M 
        indiv = group(:,r) ; 
        high =indiv(indiv~=0); 
        load_high = loading(high,:);
        var_high = var_loading(high,:);
        t_high = (load_high  - mean(load_high ,1)) ./sqrt(var_high) *sqrt(T) ; % Ngroup by K

          maxZ = nan(1000,1) ;
          for l = 1:length(maxZ) 
            Z= randn(size(t_high,1),size( sqcov,1))* sqcov ;
             maxZ(l)=  max(max( abs( Z )));
          end
           pvalue(r) = mean( maxZ > max(max( abs(t_high )))) ;
        
    end % r 

end % if if norm(group)>0

 
