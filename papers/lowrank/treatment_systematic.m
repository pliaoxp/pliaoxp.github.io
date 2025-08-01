% treatment effect
% systematic treatment
 
clear

Rep=1000;
   
   
p=400; % good to be even
n=  p+1  ;
p0=round(p/2);
p1= p-p0;
 

max_num_exceptions=  round(p0^(1/4))  ; %  max number of exceptions in time for each person in the control/treatment periods.



power= 2;



%%
%% 



gh= round((n-1)/2);
person=2; % estimate this person


  
  sigma_e= 1;
  BB=100; % true number of sieve basis
  

      W1=2*abs(ones(p,1)*randn(1,BB)); % control 
      W2=W1+2; % treatment 
      

     
      JJ=4; % max sieve dim tried
       tstat3=nan(Rep, JJ);
      
 parfor or=1:Rep
    
     %% treatment functions and noise 
 
             error=randn(n,p)*sigma_e;
             eta=unifrnd(-1,1,n,1);
             PHI= sin(eta*[1:BB]); % n by BB
            
             
            
            fgs=1./([1:BB].^power);
             sieve_coff1= W1.*(ones(p,1)*fgs);  % P BY BB
             sieve_coff2= W2.*(ones(p,1)*fgs);  % P BY BB
             theta1=PHI*sieve_coff1'; % control
             theta2=PHI*sieve_coff2'; % treatment
           
             %% treatment assignment: systematic
             X2=nan(n,p); %  treatment
             X2(:,1:p0)=zeros(n,p0);   X2(:,p0+1:p)= ones(n,p1);
             for i=1:n
                  num_exceptions= randsample(max_num_exceptions,1,0)  ;
                 mda=ones(1,p1);
                 a=randsample(p1,num_exceptions,0);
                 mda(a)=0;
                 X2(i,p0+1:p)=mda;
                 
                 mda=zeros(1,p0);
                 a=randsample(p0,num_exceptions,0);
                 mda(a)=1;
                 X2(i,1:p0)=mda;
             end
              
              
             X1=1-X2; % control 
           
             
             %% other DGP
        
          
           
             Y1=(theta1+error).*X1;  % control 
             Y2=(theta2+error).*X2; % treatment
          
             
             theta_est=nan(p,2);
             varepsilon2=nan(p,2);
              varepsilon=nan(p,2,2);
                 zeta=nan(p,2,2);
                   thetahat=nan(p,2,2);
                 
                   thetahat1 =  nan(p0,2) ;
                   zeta1 =    nan(p0,2)  ;
                    thetahat2=   nan(p1,2) ;
                     zeta2 =    nan(p1,2)  ;
                    
                     
                          
                   
    for hatJ=1:JJ            
                   
             for D=1:2
                 
                  %% sample splitting
                  
                  switch D
                      case 1 % control 
                           Y=Y1(:,1:p0); X=X1(:,1:p0);
                          p_new=p0;
                      case 2   % treatment
                           Y=Y2(:,p0+1:p); X=X2(:,p0+1:p);
                           p_new=p1;
                  end
                   g=ones(p_new,1)/p_new; % dense
    
                  by=Y;
                  bx=X;
                  by(person,:)=[]; % remove the person  th observation
                  bx(person,:)=[]; % remove the person  th observation
               
                
                   
                   for par =1:2 % samplesplitting
                       
                       lengthindividual=length(by(:,1));
                       switch par
                           case 1
                                individualindex1=[1:gh];
                                individualindex2=[gh+1:lengthindividual];
                           case 2 
                                individualindex1=[gh+1: lengthindividual ];
                                individualindex2=[1:gh]; 
                       end % switch
                       x=bx(individualindex1,:); % for regualrized 
                       y=by(individualindex1,:);   % for regualrized 
                       xc=[X(person,:); bx(individualindex2,:)];   % for iterative LS
                       yc=[Y(person,:); by(individualindex2,:)];   % for iterative LS
                       n1= length(y(:,1));
                       nc= length(yc(:,1));  % sample size of iterative LS
                       
                       
                         %% V_initial from low hatJ estimation
                         tuu =zeros(100,1);
                       for i=1:100
                             tuu(i)= norm(x.*randn(n1,p_new),2) ;
                       end
                        tuning = quantile(tuu,0.95)*1.1;
                        hB= zeros(n1,p_new);
                        diff=1;
                        tau= 0.9/max(max(x.^2)); % step size;
                        while diff>1e-7
                             hBold=hB;  
                             A=hB-tau*x.*(x.*hB-y);
                            [U2,S2,V2]=svd(A);
                             S2new=(S2-tau*tuning).*(S2>tau*tuning);
                             hB=U2*S2new*V2';
                             diff= norm(hB-hBold,'fro')^2/(p_new*n1);
                        end % while
                      %  Theta_initial = hB; 
                         [tV1,S1,tempV]=svd(hB);
                         [tempeig,tempidx]    =      sort(real(diag(S1)),'descend');
                         eigvec               =      tempV(:,tempidx); 
                         V_initial            =      eigvec(:,1:hatJ); % p by J
                         
                           %% estimate Gamma & updated V 
                         Gamma_I = zeros(nc,hatJ); 
                         V_final =zeros(p_new,hatJ);
                         for s=1:nc
                               design = (xc(s,:)'*ones(1,hatJ)).* V_initial   ; % p by J
                               Gamma_I(s,:)= yc(s,:)*design/(design'* design ); % 1 by J
                         end % s 
                         for j=1:p_new
                               design=  (xc(:,j)*ones(1,hatJ)).*  Gamma_I ;  % nc by J
                               V_final(j,:)= yc(:,j)' * design/(design'*design) ;     % 1 by J
                         end % j
                         
                          %% thetahat  and se 
                          
                           hatB= V_final'*diag(X(person,:))*V_final;
                             res= yc- xc.*(Gamma_I*V_final'); % residual, nc by p
                      
                             
                          switch D
                              case 1 % control
                                   thetahat1(:,par)= V_final* Gamma_I(1,:)' ; % p0 by 1
                                   zeta1(:,par) = (xc(1,:).*(g'*V_final/hatB* V_final') )'; %   p0  by 1 
                              case 2 % treatment
                                   thetahat2(:,par)= V_final* Gamma_I(1,:)' ; % p1 by 1
                                   zeta2(:,par) = (xc(1,:).*(g'*V_final/hatB* V_final') )'; %   p1  by 1 
                          end % switch
               
                             
                             if par==1&& D==1
                                       res_par1_D1=  res  ; % nc by p
                                    
                             end 
                             if par==1&& D==2
                                       res_par1_D2=  res   ; % nc by p
                             end 
                             if par==2&& D==1
                                      res_par2_D1= res   ; % nc by p
                             end 
                             if par==2&& D==2
                                       res_par2_D2=  res  ; % nc by p
                             end 
                           
                    
                       
                   end % par samplesplitting
                   
                     
                      
             end % D
             
               theta_est1 = mean(thetahat1,2); % control  % p0 by 1, final estimator for Theta(person,:)' 
                     theta_est2 = mean(thetahat2,2); % treatment % p1 by 1, final estimator for Theta(person,:)' 
                   
                     
             truetau=  mean(theta2(person,:)-theta1(person,:));   % scalar
             hattau=  mean(theta_est2) -mean(theta_est1)  ;   
             
            
              
                 %% se   final calcualtion
                sigmae2_par1_D1 =   diag(res_par1_D1'*res_par1_D1)/length(res_par1_D1(:,1)); % p by 1  %diag((res_par1_D1+ res_par1_D2)'*(res_par1_D1+ res_par1_D2)/length(res_par1_D1(:,1))); % p by 1
                sigmae2_par2_D1 =   diag(res_par2_D1'*res_par2_D1)/length(res_par2_D1(:,1));
                sigmae2_par1_D2 =   diag(res_par1_D2'*res_par1_D2)/length(res_par1_D2(:,1));
                sigmae2_par2_D2 =   diag(res_par2_D2'*res_par2_D2)/length(res_par2_D2(:,1));
                
                  
             %  par=1;
                 mm1=  sum(zeta1(:,1).^2.* sigmae2_par1_D1) +  sum(zeta2(:,1).^2.* sigmae2_par1_D2) ;  %sum((  zeta(:,par,1) -zeta(:,par,2)).^2.*  sigmae2_par1 );
              % par=2;
                 mm2=  sum(zeta1(:,2).^2.* sigmae2_par2_D1) +  sum(zeta2(:,2).^2.* sigmae2_par2_D2) ;   %sum((  zeta(:,par,1) -zeta(:,par,2)).^2.*  sigmae2_par2 );
             se3=sqrt( mm1/2  +mm2/2);
             
                   
             se3=sqrt( mm1/2  +mm2/2);
              
              tstat3(or, hatJ)=   (hattau- truetau)/se3;
              
     end % hatJ
             display(or)
end % or

 
 mean(abs(tstat3)< norminv(0.975),1)
 



t=[-3:0.1:3];
m=normpdf(t);
 
  
for k=1:4
    subplot(1,4,k)
    switch k
        case 1
            TITLE='power=5, N_0=10, J=1';
        case 2
             TITLE='power=5,  N_0=10, J=2';
        case 3
             TITLE='power=5, N_0=10,  J=3';
        case 4
             TITLE='power=5,  N_0=10, J=4';
    end
histogram(tstat3(:,k),'normalization','pdf')
hold on
plot(t,m,'linewidth',2,'color','k')
 
 title(TITLE,   'FontSize',16);
hold off
end
 
%% plot treatment functions and treatment effect
%{
subplot(1,2,1)
x=[-1:0.1:1]'; j=2;
% 2: treat, 1: control
     y1=sin(x*[1:BB])* sieve_coff1(j,:)'; % x b 1
     y2=sin(x*[1:BB])* sieve_coff2(j,:)' ;% x b 1
     plot(x,y1,':k',x,y2,'b','LineWidth',4)
      legend('D=0','D=1', 'FontSize',20,'Location','best');
       title('g_{D}(\eta)',   'FontSize',20);
      xlabel('\eta','FontSize',40)
      set(gca,'FontSize',20)
      
      subplot(1,2,2)
         truetau0=  (theta1 -theta2 )*g;   %  n by 1
      histogram( truetau0);
       title('Histogram of \tau_i',   'FontSize',20);
       set(gca,'FontSize',20)

subplot(1,2,1)
scatter( theta_est(:,1),theta1(person,:)')
hold on
plot([-5:5],[-5:5])
hold off

subplot(1,2,2)
scatter( theta_est(:,2),theta2(person,:)')
hold on
plot([-10:10],[-10:10])
hold off
 
 
 
t=[-3:0.1:3];
m=normpdf(t);
for k=1:4
subplot(2,2,k);
switch k
    case 1 
        tr=tstat1;
    case 2 
        tr=tstat2;
    case 3
        tr=tstat3;
    case 4
        tr=tstat4;
end
histogram(tr,'normalization','pdf')
hold on
plot(t,m,'linewidth',2,'color','k')
 TITLE='n=p=200, treatment effect';
 title(TITLE,   'FontSize',16);
hold off
end
 
 
%}

