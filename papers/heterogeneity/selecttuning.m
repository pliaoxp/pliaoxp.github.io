function [tuning, tuning0] =selecttuning(X, Y)


%% The model
% This function selects tunings for the nuclear norm penalization
% It assumes a homoskedastic Gaussian model, and uses iterations.

% Y_it = X1_it * Theta_1_ it +.... + Xd_it * Theta_d_it +   alpha_i*g_t+ v_it *stdu

 % where v_it is homoskedastic, variance =1. 

%% Inputs 

%         Y: N by T
%         X: N by T by d

%% outputs
 
%  tuning:  2 by d vector of tunings for Theta1....Thetad ; 
              %     first   row is for the first half splitting sample; the second row is for
              %         the second half splitting sample.
%         tuning0:  1 by 1  tuning for the intercept fixed effect alpha*g
       
%% first iteratively estimate the standard error , stdu


[N,T,d]= size(X);

   maxiter =   4000    ; % max iter in while for nuclear 
      whilethres= 1e-5; 
  
     for r=1:d
        tau(r)= 0.9/max(max(X(:,:,r).^2)); % step size;
     end
    
     
xxx=[];
for k=1:d
    xxx=[xxx, reshape(X(:,:,k),N*T,1)];
end
xxx=[xxx,ones(N*T,1) ];
beta=inv(xxx'*xxx)*xxx'*reshape(Y,N*T,1);
stdu=    std(reshape(Y,N*T,1)- xxx*  beta );  % 0.3298
 
  
%% tunings

for or=1:100
   for r=1:d
    tuningbfdafa(r,or)= norm(X(:,:,r).*randn(N,T),2)*1.1  ;
   end
   tuning0bfdafa(or)=norm(randn(N,T),2)*1.1 ; 
end
 

 
 
mfadfa =1;

 

while mfadfa <100
 
 
tuning0= quantile(tuning0bfdafa*stdu,0.975);   %  14.3813
for r=1:d
tuning(r)= quantile(tuningbfdafa(r,:)*stdu,0.975);   
end
 
 
  
 if  mfadfa==1
      
   hB=  zeros(N,T,d);
   hB0=  Y;
 end 
   diff=10;
   k=1;  


   while diff> whilethres && k<maxiter
    hB0old=hB0;
    hBold=hB; % intercept
   
     % hB
    for r=1:d
         
        Z=Y-(hB0+ sum(X.*hB,3)-X(:,:,r).*hB(:,:,r));
        A=hB(:,:,r)-tau(r)*X(:,:,r).*(X(:,:,r).*hB(:,:,r)-Z);
        [U2,S2,V2]=svd(A);
        S2new=(S2-tau(r)*tuning(r)).*(S2>tau(r)*tuning(r));
        hB(:,:,r)=U2*S2new*V2';
  
    end
  
     % hB0
     Z=Y-sum(X.*hB,3);
    [U1,S1,V1]=svd(Z);
    S1new=(S1-tuning0).*(S1>tuning0);
    hB0=U1*S1new*V1';
    
    Delta12=hB0-hB0old;
    Delta13=hB-hBold;
    diff= sqrt(norm(Delta12,'fro')^2/(N*T) + norm(reshape(Delta13, N,d*T),'fro')^2/(N*T*d));
    
    k=k+1;
   end % while
 
  
  residual=Y-sum(X.*hB,3)-hB0  ;
 stdu=sqrt(norm(residual,'fro')^2/(N*T));
   mfadfa =mfadfa +1;
end % mfadfa <100

 stdu;
 
 clear tuning tuning0
 %% Next, simulate tunings using two splitted sample 
 
  
  tuning=zeros(2,d);  
  for par=1:2   
      
    if par==1
        x=X(:,1:round(T-1)/2,:);
    else % par==2
         x=X(:,round(T-1)/2+1:end,:);       
    end
    % use (x,y) for low rank estimation
    
    [N,T1,d]=size(x);
    
    
     for or=1:100
        U=  randn(N,T1);
        for r=1:d
           tuningb(r,or)= norm(x(:,:,r).*U*stdu,2)*1.1  ;
        end
        tuning0b(or)=norm(U*stdu,2)*1.1 ; 
     end
     tuning01(par)= quantile(tuning0b,0.975);   %  14.3813
     for r=1:d
           tuning(par,r)= quantile(tuningb(r,:),0.975);   
     end

  clear x U
  
 
             
  end % par 
 
  tuning0=mean(tuning01);
  tuning;