function out=Cmin(Y,K,thres,matrix)

%%%%% This function is for determining the minimum constant in the threshold that 
%%%%%  guarantees the positive definiteness of POET, proposed by Fan, Liao and Mincheva (2012)
%%%%%  "Large Covariance Estimation by Thresholding Principal
%%%%%    Orthogonal Complements", manuscript of Princeton  University

%%% Model:  Y_t=Bf_t+u_t, where B, f_t and u_t represent factor loading
%%%%%        matrix, common factors and idiosyncratic error respectively.
%%%%%        Only Y_t is observable. t=1,...,n. Dimension of Y_t is p. The
%%%%%        goal is to estimate covariance matrices of Y_t and u_t.

%%%%% We apply the adaptive thresholding (Cai and Liu 2011, JASA) on either the correlation or covariance matrix of the covariance of u_t. The
%%%%% threshold takes the form: C*rate*sqrt(theta_i,j) on the (i,j)th
%%%%% entry, where rate and theta_i,j are data-driven whose explicit forms
%%%%% are given in the paper Fan, Liao and Mincheva (2012).

%%%%% Note: (1) The function out=Cmin(....)  gives the minimum constant such that, when C>out, the POET
%%%%%           estimator must be positive definite in any finite sample.
%%%%%       (2) We can run Cmin directly without running 'POET'. Cmin calls
%%%%%           'POET' itself.
%%%%%       (3) We can apply the adaptive thresholding on either the
%%%%%           correlation matrix or the covariance matrix, specified by
%%%%%           the option 'matrix'
%%%%%       (4) If no factor structure is assumed, i.e., no common factor
%%%%%         existens and var(Y_t) itself is sparse, set K=0.
 
 
 
%%% Inputs:
%%%%%  Y:    an p by n matrix of raw data, where p is the dimensionality, n
%%%%%        is the sample size. It is recommended that Y is de-meaned,
%%%%%        i.e., each row has zero mean.

%%%%%  K:    number of factors. K is pre-determined by the users. Suggestions on choosing K:
%%%%%     (1) A simple way of determining K is to count the number of
%%%%%        very spiked (much larger than others) eigenvalues of the
%%%%%        p by p sample covariance matrix of Y. 
%%%%%     (2) A formal data-driven way of determining K is described in
%%%%%        Bai and Ng (2002):
%%%%%       "Determining the number of factors in approximate factor
%%%%%       models", Econometrica, 70, 191-221. This procedure requires a
%%%%%       one-dimensional optimization.
%%%%%     (3) POET is very robust to over-estimating K. But under-estimating K
%%%%%         can result to VERY BAD performance. Therefore we strongly recommend
%%%%%         choosing a relatively large K (normally less than 8) to avoid
%%%%%         missing any important common factor.
%%%%%     (4) K=0 corresponds to thresholding the sample covariance if we believe var(Y_t) itself is sparse.

 
%%%%%  thres: the option of thresholding. Users can choose from three
%%%%%         thresholding methods:
%%%%%        'soft': soft thresholding
%%%%%        'hard': hard thresholding
%%%%%        'scad': scad thresholding
%%%%%        Details are found in Rothman et al. (2009):
%%%%%     "Generalized thresholding of large covariance matrices." JASA, 104, 177-186  


%%%%%  matrix: the option of thresholding correlation matrix. Users can
%%%%%         choose from:
%%%%%        'cor': threshold the error correlation matrix then transform back to
%%%%%               covariance matrix
%%%%%        'vad': threshold the error covariance matrix directly.



%%% Outputs:
%%%%%  out:  a positive constant.

%%% Toy example:  
%%%%%             p=100; n=100;
%%%%%             Y=randn(p,n);
%%%%%             out=Cmin(Y,3,'soft','cor');





out=fzero(@(C) mineig(Y,K,C, thres,matrix), 10);





function f=mineig(Y,K,C, thres,matrix)
%%% minieig returns the minimum eigenvalue of POET(Y,K,C,thres)
[SigmaY,SigmaU]=POET(Y,K,C, thres,matrix);
f=min(eig(SigmaU));










