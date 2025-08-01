%% this is to replicate the post-selection treatment effect model, Sec 5.3 of Fan and Liao (2022) 
%% require function " doubleselection"
clear
 Rep=1000 ; %   num of replications


r= 2 ; % true num factors. set r=0 if no factors 
 
alpha= 1  ; % strength of factors, [0,1]; % the closer to 1, the stronger 

N=200; %   dim
bigR=10;

   
 
T= N+1;

 

      
      
 %% generate parameters 
 
 
  
 
varf=2; % std of factors
 varu=1; % std of U
 nu=zeros(N,1);  nu(1)=1; nu(2)=-1.5; nu(3)=0.5; 
 theta=nu;  % sparse vector 
 beta= 1;  %  treatment effect to estimate 
 
 
 
 
 
% tuning:
lambday= 1.5*sqrt(beta^2+1)*sqrt(log(N)/T);
lambdag= 1.5*sqrt(log(N)/T);

 
blocksize= 5;
block = nan(blocksize,blocksize); 
for i=1:blocksize
    for j=1:i
        block(i,j)=0.5^(abs(i-j));  block(j,i)= block(i,j);
    end
end
halfblock=block^(1/2) * varu; 
  % Sigmau
       halfSigmaU=eye(N);
       halfSigmaU(1:blocksize,1:blocksize)=halfblock ;
       halfSigmaU(blocksize+1:blocksize*2,blocksize+1:blocksize*2 )=halfblock ;
       halfSigmaU(N-(blocksize-1):N,N-(blocksize-1):N)=halfblock ;
  
  
          %% generate  loading , with weak factors 
  
   Z=sin(randn(N,1)) ;
 
 
 if r>0
L=  Z.^[1:r] +2+randn(N,r)*0.5 ; 
 else 
     L=zeros(N,r);
 end
 

 L = L *N^(-(1-alpha)/2)  ;
 
 
 

     %% start loop 
  
       z=nan(Rep,4);
       zdouble=nan(Rep,1);    
       zpca=nan(Rep,1);  

       parfor or=1: Rep

       warning('off','all')
          
        
           %% DGP 
           U=halfSigmaU*randn(N,T);
           eta= randn(T,1);
           epsilong= randn(T,1);
           
               F=randn(T,r)* varf;
               X=L*F'+ U;   % N x T
            
           G= X'*theta+epsilong ; % T by 1
           Y= G*beta + X'*nu +eta; % T bY 1
            
          
           %% inference 

           r0= max(r,1);  
              
           bigW = X(:,1).^[1:bigR];   % initial transformation 
           for mm=1:4
               R= r0+mm-1;
                 [ betahat , se ]=   doubleselection(R, X,Y, G, bigW, lambday , lambdag)
               z(or, mm)=( betahat-beta)/se ;
           end % R
           
           %% Double lasso, R=0 
            
            [ betahatdouble, sedouble]=   doubleselection(0, X,Y, G, [], lambday , lambdag) ;
                zdouble(or)=(betahatdouble -beta)/sedouble ;
                
                %% PCA R =r 
               
               [V,d]= eigs(X*X',r);
               hatloading= V*sqrt(N); 
                 [ betahat , se ]=   doubleselection(r, X,Y, G,    hatloading , lambday , lambdag)
            zpca(or)=(betahat -beta)/se  ;
                
               if mod(or,200)==0
                   or 
               end
           
       end % or
       
    
t=[-3:0.1:3];
normal=normpdf(t);
 
   for k=1:6
       subplot(2,3,k);
       switch k
       case 1
           if r>0
           TITLE='DP R=r';
           else % r=0
                TITLE='DP R=1, true r=0';
           end 
            histogram(z(:, k),'normalization','pdf')
       case 2 
           if r>0
           TITLE= 'DP R=r+1';
           else % r=0
                TITLE='DP R=2, true r=0';
           end 
            histogram(z(:, k),'normalization','pdf')
       case 3 
           if r>0
           TITLE= 'DP R=r+2';
          else % r=0
                TITLE='DP R=3, true r=0';
           end 
            histogram(z(:, k),'normalization','pdf')
        case 4 
            if r>0
           TITLE= 'DP R=r+3';
            else % r=0
                TITLE='DP R=4, true r=0';
           end 
            histogram(z(:, k),'normalization','pdf')

       case 5
           TITLE='double selection';
           histogram(zdouble,'normalization','pdf')

         case 6
             if r>0
                 TITLE='PCA R=r ';
             else 
                   TITLE='double selection ';
             end
           histogram(zpca,'normalization','pdf')
       end
      
       hold on 
       plot(t,normal,'LineWidth', 3)
        title(TITLE,   'FontSize',16);
       hold off
   end
   


 

  