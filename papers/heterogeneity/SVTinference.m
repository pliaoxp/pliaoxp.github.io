
function [ theta, ster ,grouptheta , stergroup] = SVTinference(Y,X,Common, time,  K, K0,tuning,  tuning0, group)

%% The model
% Y_it = X1_it * Theta_1_ it +.... + Xd_it * Theta_d_it +   alpha_i*g_t+ u_it
% Theta_r_it = lambda_r_i* f_r_t 


%
%% Inputs:
%         Y: N by T

%         X: N by T by d

%         Common: N by T by d , common components in X, we have
              % X_it= Common_it + E_it
              % If Common_it = l_i’*w_t, i.e., X_it follows  factor
              % structres, can ue the following code to estimate Common
              
              %{
              % input the number of factors Kx for each regressor in X
              [N,T,d]=size(X);;
              for r=1:d
                    Z= X(:,:,r);
                    Sam_cov= Z*Z'/T; 
                    [tempV,tempD]         =      eig(Sam_cov/N);   
                    [tempeig,tempidx]     =      sort(real(diag(tempD)),'descend');
                    eigvec                =      tempV(:,tempidx); 
                    u1                    =      eigvec(:,1:Kx(r));
                    l                     =      u1*sqrt(N);  % N by Kx(r)
                    w                     =      Z'*l/N; % T by Kx(r)
                    Common(:,:,r)         =      l*w';
                    clear Z Sam_cov tempV tempD tempeig tempidx eigvec u1 l w 
              end
              %}
              
              
%         time: the time "t" at which we make inference about Theta_d_it

%         K: d by 1, number of factors in each Theta 
%         K0: 1 by 1, number of factors in intercept

%         tuning:  2 by d vector of tunings for Theta1....Thetad ; first
              %         row is for the first half splitting sample; the second row is for
              %         the second half splitting sample.
%         tuning0:  1 by 1  tuning for the intercept fixed effect alpha*g
          
              %    Use the following code to calculate the tunings first:
              
              %{
              [tuning, tuning0] =selecttuning(X, Y);
              %}
              
%         group: R by M:  R is the number of individuals in each group, M
              %           is the number of groups, whose average effects are to estimate. 
              %     Each column of group is indices of the cross sections for each group. 
              %     If the group sizes are  not the same, R is the max group size, and all other groups put zeros
              %     For instance, to estimate two groups:
              %           Group 1: individuals {1,3};
              %           Group 2: individuals {2,5,6}. 
              %        Then group= [  1,2;3,5;0,6  ], which is 
              % 1 2
              % 3 5
              % 0 6
              % and R=3
              
              % set group =0 if do not estimate any group average effect


%% Outputs:  Columns of the following output variables correpond to
             % individual regressors of X, at t= time
             
%             theta:   N by d, the estimated theta 
%              ster:   N by d, standard error
%       groupthetea:   M by d, the averaged group effects, if group=0 then
%                              groupthta=0
%         stergroup:   M by d, group standard errors, if group =0 then stergroup=0
%    
  


%% Sample code

%{

d=2;
N=100;
T=N;
sigma=2;
for r=1:d
    Theta(:,:,r)=(randn(N,1)+2)*(randn(1,T)+3);
end

    Kx=[1,1];
    K=[1,1];
    K0=1;
    for r=1:d
       X(:,:,r)= randn(N,1)*randn(1,T)+randn(N,T); 
    end
    Y=sum(Theta.*X,3) +randn(N,1)*randn(1,T)+randn(N,T)*sigma;
     [N,T,d]=size(X);
              for r=1:d
                    Z= X(:,:,r);
                    Sam_cov= Z*Z'/T; 
                    [tempV,tempD]         =      eig(Sam_cov/N);   
                    [tempeig,tempidx]     =      sort(real(diag(tempD)),'descend');
                    eigvec                =      tempV(:,tempidx); 
                    u1                    =      eigvec(:,1:Kx(r));
                    l                     =      u1*sqrt(N);  % N by Kx(r)
                    w                     =      Z'*l/N; % T by Kx(r)
                    Common(:,:,r)         =      l*w';
                    clear Z Sam_cov tempV tempD tempeig tempidx eigvec u1 l w 
              end
    time=1;
    group=[1;3;4];       
    [tuning, tuning0] =selecttuning(X, Y);  
    [theta, ster ,grouptheta , stergroup] = SVTinference(Y,X,Common, time,  K, K0,tuning,  tuning0, group);
    
 
%}

%% code
   
    [N,T,d]=size(X);
    [R, M]=size(group);
    toler= 1e-4 ;
    E= X- Common;
    maxiter =   2000    ; % 2000 max iter in while for nuclear 
   
   
   
   
    
   %% Sample splitting 
        
           
    by=Y;
    bx=X;
    be=E;
    bcommon=  Common;
    by(:,time)=[]; % remove the time  th observation
    bx(:,time,:)=[]; % remove the time  th observation
    be(:,time,:)=[]; % remove the time  th observation
    bcommon(:,time,:)=[]; % remove the time  th observation
    

    
 
  for par=1:2
     
  
      
    if par==1 
       % use (x,y) for low rank estimation;
       % use (xc, ec, yc) for post low rank least square inference
        x=bx(:,1:round(T-2)/2,:);
        y=by(:,1:round(T-2)/2); 
        xc=[X(:,time,:), bx(:,round(T-2)/2+1:end,:)];
        ec=[E(:,time,:), be(:,round(T-2)/2+1:end,:)];
        yc=[Y(:,time),by(:,round(T-2)/2+1:end)]; % N by Tc
        commonC=[Common(:,time,:), bcommon(:,round(T-2)/2+1:end,:)];
         T1= length(y(1,:));
    else % par==2
         x=bx(:,round(T-2)/2+1:end,:);
         y=by(:,round(T-2)/2+1:end); 
         xc=[X(:,time,:),bx(:,1:round(T-2)/2,:)];
         ec=[E(:,time,:),be(:,1:round(T-2)/2,:)];
         yc=[Y(:,time),by(:,1:round(T-2)/2)]; % N by Tc
         commonC=[Common(:,time,:),bcommon(:,1:round(T-2)/2,:)];
          T1= length(y(1,:));
    end
       
       for r=1:d
        tau(r)= 0.9/max(max(x(:,:,r).^2)); % step size;
        end
   
   
     
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
   
   
   %% check if any Theta matrix or the interactive effect Theta0 matrix is zero
       for r=1:d
           if norm(hB(:,:,r),'fro')/sqrt(N*T1)<1e-5 %  zero, no factors
          K(r)=0;
           end
       end
       if  norm(hB0,'fro')/sqrt(N*T1)<1e-5
           K0=0;
       end
   
       
       %% extract eigenvectors from the low rank matrices
       
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
         tLambda0=U0*sqrt(N) ; % N by K0
       
           
    clear hB hB0  hBold    hB0old  Z A U2 S2 V2 S2new    U1 S1 V1 S1new Delta12 Delta13 
    clear tempV S1 V1 tempeig tempidx   eigvec U1 U0 
    clear newyc newxc  
    
    
  
 
    
          %% obtain initial estimates, to estimate the partial out components 
 
    
    
    Ic= length(yc(1,:)); 
      
   for s=1:Ic
       dm=[];
       for r=1:d
           dm=[dm,(xc(:,s,r)*ones(1,K(r))).* tLambda1(:,1:K(r),r)];
       end
       
           design= [tLambda0,dm];
            ols= inv(design'*design)*design'*yc(:,s);
           tcons_factor(:,s)= ols(1:K0); % correp constant  , Tcx 1
           tfactor1(:,s)= ols(K0+1:end);  % corresp X   , arranged as r=1, r=2,... r=d
        
       
       clear dm design ols  
   end
 
       
    % loadings
    
      
    
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
     
   
   
      pp=0;
    for r=1:d
        if K(r)>0
      thetainitial(:,:,r)=tloading1(pp+1: pp+K(r) ,:)'*tfactor1(pp+1: pp+K(r),:) ; % N by Ic
        else  thetainitial(:,:,r)=zeros(N, Ic); 
        end
         pp=pp+K(r);
    end
    newyc=yc-sum( commonC.*thetainitial, 3); 
    newxc=ec;
  
  %% inference
   
   % factors
      
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
       
       % residual 
       u(i,:) = newyc(i,:) -     ols' * design' ;   % N by T
       
       clear dm design ols  
   end
   
   %% estimate theta for inference
    pp=0;
    
   
     
     
   for r=1:d
        if K(r)>0
      thetahat(:,r, par)=loading2(pp+1: pp+K(r) ,:)'*factor2(pp+1: pp+K(r),1) ; % N by 1 of estimated theta at t =time
      
        
      %% individual  standard error
       w= ec(:,:,r).* u; 
       Se= diag(diag( ec(:,:,r)*ec(:,:,r)'/Ic)); % N by N
       Sw= diag(diag( w*w'/Ic));  % N by N
       Vlambda1  =  loading2(pp+1: pp+K(r) ,:)* Se* loading2(pp+1: pp+K(r) ,:)'/N; % num factor by num factor
       Vlambda2  =  loading2(pp+1: pp+K(r) ,:)* Sw* loading2(pp+1: pp+K(r) ,:)'/N;
       Vlambda = inv( Vlambda1)*  Vlambda2 * inv( Vlambda1); % num factor by num factor
       
       sel =diag(loading2(pp+1: pp+K(r) ,:)'* Vlambda *  loading2(pp+1: pp+K(r) ,:) ) /N   ;  % N by 1 % lambda_i'*V_lambda* lambda_i/N
       % homoske
       Sf =  factor2(pp+1: pp+K(r),:)* factor2(pp+1: pp+K(r),:)'/ Ic;
       daf = factor2(pp+1: pp+K(r),1)'* inv(Sf) * factor2(pp+1: pp+K(r),1);  % 1 by 1
       ofafa =  ( diag(Sw)./(diag(Se).^2)); % N by 1
       sef =  daf*  ofafa /Ic; % N by 1,  f_t'V_f f_t/ T
       
       se(:, r, par)=sqrt(sel+sef); % N by 1, std error
     
     
      
      %% group standard error
        
      if norm(group)>0 % want to estimate group 
      %    groupster:  M by d, group standard error
         for g=1: M
             G= sum(group(:,g)>0) ; %   size of group g
             membership= group(1:G ,g);
             barlambda=mean(loading2(pp+1: pp+K(r) ,membership),2); % num factor by 1
             selgroup = barlambda'*Vlambda* barlambda/N; % scalar 
             sefgroup=daf* mean(ofafa(membership) ) /(Ic*G); % scalar 
             segroup(g,r ,par)=sqrt(selgroup+sefgroup); % group standard error
          
           clear membership barlambda G
         end % g
      
      end % group
      
        else  thetahat(:,r, par)=zeros(N, 1);  %  K(r)==0
            se(:, r, par)=zeros(N, 1); 
             segroup(:,r ,par)=zeros(M,1);
        end % if K(r)>0
        
      
       pp=pp+K(r);
      
      clear   Vlambda1 Vlambda2 w   Vlambda Sf Se Sw daf sef sel
   end % r=1:d
    
   
    %%   fix effect
      
      fixeffect= cons_loading'* cons_factor(:,1);  % N by 1
      
      
   
   clear u w; 
  
  
   clear tcons_factor tfactor1   tloading1  tcons_loading tLambda1 tLambda0 
   clear newyc newxc yc xc Ic  loading2  factor2  thetainitial
   clear  cons_factor    cons_loading  
   clear   x y   ec   commonC T1 w u       
   
 end % par    
   
 theta= (thetahat(:,:, 1) +thetahat(:,:, 2))/2; % N by d
 for m=1:M
     index= group(:,m);
     index(group(:,m)==0)=[];
     grouptheta(m,:) = mean(theta(index,:));
     clear index
     
 end
 
   ster= (se(:, :, 1)+ se(:, :, 2))/2; % N by d, standard error
    if norm(group)>0
       stergroup= (segroup(:,: ,1)+ segroup(:,: ,2))/2; % M by d group standard error
    else stergroup=0;
    end
  
    % outputs: theta, grouptheta, ster, stergroup
    