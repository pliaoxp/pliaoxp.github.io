% The model
% Y_it = X1_it * Theta_1_ it +.... + Xd_it * Theta_d_it +   alpha_i*g_t+ u_it
% Theta_r_it = lambda_r_i* f_r_t 
% Xr_it = l_r_i * w_r_t + e_r_it,  r= 1...d
% Assume e_r_it 's are independent across r and t. 

%tu
%
% Inputs:
%         Y: N by T
%         X: N by T by d
%         time: the time "t" at which we make inference about Theta_d_it
%         tuning:  d by 1 vector of tunings for Theta1....Thetad 
%         tuning0:  1 by 1  tuning for the intercept fixed effect alpha*g
%       Common :  common component in X , 
%        Keffect: % number of factors in slope 


% Outputs:
%        theta: N by d, the estimated theta at t=time, for all X
%         ster: N by d ,     standard error

 

function [theta,    ster ,stergroup, fixeffect ,  beta_final,  var_beta_final, Corrmatrix] = factorslopeestimate(Y,X, Common , Keffect, time,   tuning,  tuning0, group)

[N,T,d]=size(X);
 K = ones(d,1) * Keffect;  % number of factors in slope 


 beta = nan(N,Keffect,2);  % loading of theta of the first X
     var_beta   = nan(N,Keffect,2);  % variance of beta 
   segroup= nan(size(group,2),d,2) ;
Corrma =nan(Keffect,Keffect,2) ;



     maxiter =   4000    ; % 2000 max iter in while for nuclear 
    % tau=  0.9*1/max(max(max(X.^2))); % step size
     toler= 1e-4 ;
     tau=nan(d,1) ;
        for r=1:d
        tau(r)= 0.9/max(max(X(:,:,r).^2)); % step size;
        end
    
   %% Estimate the factor structures in each X
    
 E  = X -Common;

  
   
     %% estimate residuals :
     % for one option of se 
     %  for estimating rank
     % for compare with nuclear norm penalization only estimator
      
   hB= zeros(N,T,d);
   hB0=  Y;
   diff=1;
   k=1;  


   while diff>toler && k<maxiter
    hB0old=hB0;
    hBold=hB; % intercept
    % hB
    
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
   
   residual3=Y-hB0-sum(hB.*X,3);

 
  
        bresidual3 =residual3;
       bresidual3(:,time)=[];

 %% estimate number of factors in interactive effect  
  
   [tempV,S1,V1]=svd(hB0);
        [tempeig,tempidx]     =      sort(real(diag(S1)),'descend');
 Cthresh= 0.1; 
                 thrs = sqrt(tuning0* max(tempeig)) ; 
                a=sum(tempeig> thrs*  Cthresh) ;
               K0= min(max(a,1), 2  );  
       
           %{
                 Kbar = 2; % no more than 2 factors in interactive effect 
             K=nan(d,1); 
                  Cthresh= 0.1; 
                  for r= 1:d
                         [tempV,S1,V1]=svd(hB(:,:,r));
                        [tempeig,tempidx]     =      sort(real(diag(S1)),'descend');
                        thrs = sqrt(tuning(r)* max(tempeig)) ; 
                        a=sum(tempeig> thrs* Cthresh) ;
                        K(r)= min(max(a,1), Kbar  ) ; 
                  end % r 
                  hatK= K; 
           %}
      
     
   %% Sample splitting 
        
    by=Y;
    bx=X;
    be=E;
    bcommon=  Common;
    by(:,time)=[]; % remove the time  th observation
    bx(:,time,:)=[]; % remove the time  th observation
    be(:,time,:)=[]; % remove the time  th observation
    bcommon(:,time,:)=[]; % remove the time  th observation
    
    
   thetahat= nan(N,d,2) ;
   
      
 
        varf3=zeros(N,d,2);
       varl3=zeros(N,d,2);
       
 
       gh= round((T-1)/2);

     
  for par=1:2
     


          lengthtime=length(by(1,:)); 
   
         switch par
          case 1
              timeindex1=[1:gh];
               timeindex2=[gh+1:lengthtime];
          case 2 
              timeindex1=[gh+1: lengthtime ];
              timeindex2=[1:gh]; 
      end % switch
      
        
      x=bx(:,timeindex1,:);
      y=by(:,timeindex1); 
      xc=[X(:,time,:), bx(:,timeindex2,:)];
      ec=[E(:,time,:), be(:,timeindex2,:)];
      yc=[Y(:,time),by(:,timeindex2)]; % N by Tc
      commonC=[Common(:,time,:), bcommon(:,timeindex2,:)];
       T1= length(y(1,:));
      Ic= length(yc(1,:)); 

        % u= [residual(:,time),bresidual(:, timeindex2) ];
  
       uuu= [residual3(:,time),bresidual3(:, timeindex2) ];
       
       

          %% low rank estimation
    
   hB= zeros(N,T1,d);
   hB0=  y;
   diff=1;
   k=1;  


   while diff>toler && k<maxiter
    hB0old=hB0;
    hBold=hB; % intercept
    % hB
    
     % hB
    for r=1:d
         
        Z=y-(hB0+ sum(x.*hB,3)-x(:,:,r).*hB(:,:,r));
        A=hB(:,:,r)-tau(r)*x(:,:,r).*(x(:,:,r).*hB(:,:,r)-Z);
        [U2,S2,V2]=svd(A);
        S2new=(S2-tau(r)*tuning(r)).*(S2>tau(r)*tuning(r));
        hB(:,:,r)=U2*S2new*V2';
  
    end
    % hB0
     Z=y-sum(x.*hB,3);
    [U1,S1,V1]=svd(Z);
    S1new=(S1-tuning0).*(S1>tuning0);
    hB0=U1*S1new*V1';
   
 
    
    Delta12=hB0-hB0old;
    Delta13=hB-hBold;
    diff= sqrt(norm(Delta12,'fro')^2/(N*T1) + norm(reshape(Delta13, N,d*T1),'fro')^2/(N*T1*d)); 
    
    k=k+1;
   end % while
   %% get eigenvectors
   
    if k +2  < maxiter
        converg(par)=1;
    else converg(par)=0;
    end
   % disp(k);
   
   
       for r=1:d
           [tempV,S1,V1]=svd(hB(:,:,r));
            [tempeig,tempidx]     =      sort(real(diag(S1)),'descend');
               eigvec                =      tempV(:,tempidx); 
                U1                    =     [ eigvec(:,1:K(r)), zeros(N, max(K)-K(r))];
           tLambda1(:,:,r)=U1*sqrt(N) ; % N by Kr  corres to X
       end
        [tempV,S1,V1]=svd(hB0);
        [tempeig,tempidx]     =      sort(real(diag(S1)),'descend');
         eigvec                =      tempV(:,tempidx); 
         U0                    =        eigvec(:,1:K0) ;
         tLambda0=U0*sqrt(N) ; % N by K0  corres to X
 
           
    clear hB hB0  hBold    hB0old  Z A U2 S2 V2 S2new    U1 S1 V1 S1new Delta12 Delta13 
    clear tempV S1 V1 tempeig tempidx   eigvec U1 U0 
    clear newyc newxc  
    
    
  
 
    
    
    %%    estimate partial out  
     

    % mineig_check(par)=0;  % check if ill conditioned design
    % eig_threshold = 1e-7;
    Ic= length(yc(1,:)); 
      
      tcons_factor= nan(K0,Ic); 
      tfactor1 = nan(sum(K),Ic);  %   dim(design) = K0+ sum(K)
      % dim(dm) = sum(K)
   for s=1:Ic
       dm=[];
       for r=1:d
           dm=[dm,(xc(:,s,r)*ones(1,K(r))).* tLambda1(:,1:K(r),r)];
       end
        design= [tLambda0,dm];
        ols= inv(design'*design)*design'*yc(:,s);
        tcons_factor(:,s)= ols(1:K0); % correp constant  , Tcx 1
       tfactor1(:,s)= ols(K0+1:end);  % corresp X   , arranged as r=1, r=2,... r=d
       %  if min(eig(design'*design))<eig_threshold
        %     mineig_check(par)=1;  % then ill posed
      %  end

    
       clear dm design ols  
   end
 
       
    % loadings
    
     tcons_loading= nan(K0,N); 
         tloading1 = nan(sum(K),N); 
    
   for i=1:N
       dm=[];
       pp=0;
       for r=1:d
           dm=[dm,(xc(i,:,r)'*ones(1,K(r))).* tfactor1(pp+1: pp+K(r)   ,:)']; % T by ...
            pp=pp+K(r);
       end
        design= [tcons_factor',dm];
        ols= inv(design'*design)*design'*yc(i,:)';
        tcons_loading(:,i)= ols(1:K0); % correp constant  , 
       tloading1(:,i)= ols(K0+1:end);  % corresp X   , arranged as r=1, r=2,... r=d
      
        clear dm design ols  
   end
     

   
    %% partial our estimator's inference
      pp=0;
        thetainitial= nan(N,length(yc(1,:)),d) ; 
    for r=1:d
      thetainitial(:,:,r)=tloading1(pp+1: pp+K(r) ,:)'*tfactor1(pp+1: pp+K(r),:) ; % N by Tc
      pp=pp+K(r);
    end
    newyc=yc-sum( commonC.*thetainitial, 3); 
    newxc=ec;
  
 
   
   % factors
       

    cons_factor=  nan(K0,Ic); 
     factor2= nan(sum(K),Ic);

   for s=1:Ic
       dm=[];
       for r=1:d
           dm=[dm,(newxc(:,s,r)*ones(1,K(r))).* tLambda1(:,1:K(r),r)];
       end
        design= [tLambda0,dm];
        ols= (design'*design)\design'*newyc(:,s);
        cons_factor(:,s)= ols(1:K0); % correp constant  , Tcx 1
       factor2(:,s)= ols(K0+1:end);  % corresp X   , arranged as r=1, r=2,... r=d
        
       clear dm design ols  
   end
  % loadings

    cons_loading=  nan(K0,N); 
       loading2= nan(sum(K),N); 
 
 

   for i=1:N
       dm=[];
       pp=0;
       for r=1:d
           dm=[dm,(newxc(i,:,r)'*ones(1,K(r))).* factor2(pp+1: pp+K(r)   ,:)']; % T by ...
            pp=pp+K(r);
       end
        design= [cons_factor',dm];
        ols= (design'*design)\design'*newyc(i,:)'; % ... by 1
        cons_loading(:,i)= ols(1:K0); % correp constant  , 
       loading2(:,i)= ols(K0+1:end);  % corresp X   , arranged as r=1, r=2,... r=d
       
   
      
             
        
       clear dm design ols  
   end % i 
    
     %% estimate theta
    pp=0;
     
    %  thetahat has been defined earlier 
   for r=1:d
      thetahat(:,r, par)=loading2(pp+1: pp+K(r) ,:)'*factor2(pp+1: pp+K(r),1) ; % N by 1 of estimated theta at t =time
        beta(:,:,par)= loading2(1: K(1) ,:)'; % N by K 
      pp=pp+K(r);
   end % d
 
 
  
 
 
   %% standard error   

    
    pp=0;
    for r=1:d
  
      
   Vlambda1  =  loading2(pp+1: pp+K(r) ,:)* diag(ec(:,1,r).^2)* loading2(pp+1: pp+K(r) ,:)'/N;
      Se= diag( ec(:,:,r)*ec(:,:,r)'/Ic); % N by 1
       Sf= factor2(pp+1: pp+K(r),:)*factor2(pp+1: pp+K(r),:)'/Ic;
        
       % full sample
      
       w= ec(:,:,r).* uuu; 
       
       Sw= diag( w*w'/Ic);  % N by 1
       Vlambda2  =  loading2(pp+1: pp+K(r) ,:)*  diag(w(:,1).^2)* loading2(pp+1: pp+K(r) ,:)'/N;
       Vlambda = inv( Vlambda1)*  Vlambda2 * inv( Vlambda1);
       varl3(:, r, par) =diag(loading2(pp+1: pp+K(r) ,:)'* Vlambda *  loading2(pp+1: pp+K(r) ,:) )   ;  % N by 1 % lambda_i'*V_lambda* lambda_i 
      varf3(:, r, par) = factor2(pp+1: pp+K(r),1)'  *inv(Sf) * factor2(pp+1: pp+K(r),1)*   (   Sw./(Se.^2)  ) ;% N by 1,  f_t'V_f f_t 
      
     
   
        %% group standard error
        
        dSe= diag( ec(:,:,r)*ec(:,:,r)'/Ic); % N by 1
          dSw= diag( w*w'/Ic);  % N by 1
        homow=  dSw./(dSe.^2) ; % N by 1


      if norm(group)>0 % want to estimate group 
      %    groupster:  M by d, group standard error
        [GG, M]=size(group);
         for g=1: M
             G= sum(group(:,g)>0) ; %   size of group g
             %membership= group(1:G ,g);
             membership = group(group(:,g) ~= 0, g);
             barlambda=mean(loading2(pp+1: pp+K(r) ,membership),2); % num factor by 1
 
            
             selgroup = barlambda'*Vlambda* barlambda/N; % scalar  old
            daf = factor2(pp+1: pp+K(r),1)'* inv(Sf) * factor2(pp+1: pp+K(r),1);  % 1 by 1 % old 
             sefgroup=daf* mean(  homow(membership) ) /(Ic*G); % scalar  old
          segroup(g,r ,par)= selgroup+sefgroup ;  
            

           clear membership barlambda G
         end % g
     
      
      end % group
      
       %% beta variance, for test group homogeneity 
        if r==1 % focus on first regressor for testing homo effect
             
           S1= inv( Sf ); 
           S2 = diag(S1);
           Corrma(:,:,par) =  diag(sqrt(1./S2))*S1* diag(sqrt(1./S2)) ; % used for within group homo test
         
        var_beta(:,:,par) =  homow *  diag(S1)'; % N by K % used for within group homo test
       end 
   
      pp=pp+K(r);
      
      
      
      
  end % d
 

   
    %%   fix effect
      
      fixeffect= cons_loading'* cons_factor(:,1);  % N by 1
     
    
   
   clear u w; 
  
  
   clear tcons_factor tfactor1   tloading1  tcons_loading tLambda1 tLambda0 
   clear newyc newxc yc xc Ic  loading2  factor2  thetainitial
   clear  cons_factor    cons_loading  
   clear   x y   ec   commonC T1 w u       
   
 end % par    

 
 theta= (thetahat(:,:, 1) +thetahat(:,:, 2))/2; % N by d
    beta_final = (beta(:,:,1) + beta(:,:,2))/2; % the loading of the coeff for first X. 
     var_beta_final =(var_beta(:,:,1)+ var_beta(:,:,2)  ) /2; % variance of beta_final


  
  
     %% standard error  

    Ic = round((T-1)/2)+1 ; 
     
   
         % full sample method 4
    
        
      fullvar_est=  varl3(:,:,1)/N + varf3(:,:,1) /Ic+varl3(:,:,2)/N + varf3(:,:,2) /Ic...
              +  varl3(:,:,1)/N+ varl3(:,:,2)/N; % N by d
         ster=  1/2* sqrt(fullvar_est) ;    % N by d , standard error
     
     

          %%  group se 
 if norm(group)>0

         stergroup =   sqrt((segroup(:,: ,1)+ segroup(:,: ,2))/2); %   old  M by d group standard error, M= num groups

    else stergroup=0;
 end  % if norm(group)>0


 Corrmatrix= mean(Corrma,3);