%% this is to replicate the forecast model, Sec 5.2 of Fan and Liao (2022) 
%% require a function ``forecastmethod"


clear
 bigR=10; 
 r=2;  % true num factors 
  N=100; %   dim
 
    alpha= 0.2  ;% [0,1]   % strength of factors, the closer to 1 the stronger
 rho=0.9    ; % serial correlations in the noise

  
TT= 150 ;  % total number: in and out of sample
 
Tcandidate = [50, 100]; % candidate  samplesize 

 

% covariancae of noise 
 %{
blocksize= 5;
block= nan(blocksize,blocksize); 
for i=1:blocksize
    for j=1:i
        block(i,j)=0.7^(abs(i-j));  block(j,i)= block(i,j);
    end
end
halfblock=block^(1/2);
SigUhalf=    kron(eye(N/blocksize), halfblock);
 %}

Rep =50 ; 


   char = nan(length(Tcandidate) , 4, Rep);
   roll = nan(length(Tcandidate) , 4, Rep);
   init =  nan(length(Tcandidate) , 4, Rep); 
 
   parfor or = 1:Rep 

 
  %% generate loading : depending on "characteristics"
      
 
  %  character Weight
   Z=sin(randn(N,1)) ; % loading characteristic
  bigwchar=  Z.^[1:bigR];
  ter= Z.^[1:r] +2+randn(N,r)*0.5  ;  
L= ter*N^(-(1-alpha)/2) ; % loading
        

  
 %% generate "big data "
 
   F=randn(TT,r);
 UU =nan(N,TT); 
        UU(:,1)=  randn(N,1)  ;  %randn(N,1);
    for t=2:TT
        UU(:,t)= UU(:,t-1)*rho+randn(N,1);
    end 
      
 

       YY=L*F'+   UU;   % N x TT "big data" to estimate factors 
 
        %% generate outcome to forecast 
       XX=zeros(TT,1);  % outcome variable to forecast  
 XX(1)=0; 
 delta=randn(r+2,1);
  delta(r+1)= 0.5; % dynamic must be less than one
delta(r+2) =1.5; % intercept
for t=1:TT-1   
    XX(t+1)=  F(t,:)*delta(1:r)  +XX(t)* delta(r+1)+ delta(r+2)+randn;
end;

%% some weights to use 
 % rolling window Weight/ preavailable data 

 % assume that previously, we have a data "Y0" of size N by Tprevis. 
 % assume the  previous loading is correlated with the true loading

 Tprevis= 40; 
 previous_load= 0.8*L+randn(N,r)*0.5 ; 
 Y0=previous_load*randn(r, Tprevis)+randn(N, Tprevis);
 [a,b]= eigs(Y0*Y0',bigR);
  m=a*sqrt(N);
  epsilon=1/5;
  bigwrolling=  m./(ones(N,1)*max(max(abs(m))*epsilon,1));
  
   

%% start loop forecasting



 ratio_char= nan(length(Tcandidate) , 4);
 ratio_roll= nan(length(Tcandidate) , 4);
 ratio_initi= nan(length(Tcandidate) , 4);
 
 
for t =1:length(Tcandidate) 

    T= Tcandidate(t) ;  % samplesize 
 errorchar= nan(TT-T,4); 
  errorrolling = nan(TT-T,4); 
  errorinitial= nan(TT-T,4); 
  errorpca= nan(TT-T,1); 

for i=1:TT-T
    %%   data
     Y=YY(:,i:i+T-1); %N by T, "big data" to estimate factors 
      X= XX(i:i+T-1);  % T by 1 % outcome variables 
    xnew=XX(i+T);% true outcome to forecast 
  
       
    %% forecast 

          for R=r:r+3 

               % character
           forecast=forecastmethod(  R, Y, X, Z.^[1:bigR]  ) ; 
            errorchar(i, R-r+1)=(  forecast -xnew)^2;

              %  rolling window
             forecast=forecastmethod(  R, Y, X, bigwrolling ) ; 
            errorrolling(i, R-r+1)=(  forecast -xnew)^2;
            
            %  initial transformation
     
             forecast=forecastmethod(  R, Y, X,  Y(:,1).^[1:bigR]  ) ; 
            errorinitial(i, R-r+1)=(  forecast -xnew)^2;
            
         end % R
    
      
    
    % PCA % use true number of factors r
    
    [V,d]=eigs(Y*Y'/(N*T),r);
    beta = V*sqrt(N); 
     forecast=forecastmethod(  r, Y, X,  beta ) ; 
    errorpca(i)=(  forecast -xnew)^2;
    
     
      
    
end % i 


%% characteristic weight 

   ratio_char(t,:)= mean(  errorchar,1)/mean(errorpca)  ; 
  ratio_roll(t,:)= mean(  errorrolling,1)/mean(errorpca) ; 
   ratio_initi(t,:)= mean(  errorinitial,1)/mean(errorpca) ; 
     
      
     
  end % t = Tcandidate 

   char(:,:,or) =  ratio_char;
   roll(:,:,or) =  ratio_roll; 
   init(:,:,or) =  ratio_initi ; 
 

   end % or 

 characweight=  mean(char,3) 
 rollwinweight=  mean(roll,3)
 initilweight=  mean(init,3)

