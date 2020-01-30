% ************************************************************************* %
%                                                                           %
%    ** Partial Least Squares - Structural Equation Modeling K-Means **     %
%                                                                           %
% ************************************************************************* %

% Authors: Mario Fordellone and Maurizio Vichi 2018
% Sapienza University of Rome
% mail: mario.fordellone@uniroma1.it / maurizio.vichi@unioroma1.it

% ************************************************************************* %

function [A_ott, LV_ott, Y_ott, B_ott, R2_ott, com_ott, GoF_ott, U_ott, e_ott, Evolution_e_ott, iter_ott, GoF_adj_ott,e2] = PLS_SEM_KM(MV,DP,DB,K,Mmode,lv_lab)

% The function estimates a Structural Equation Model with Partial Least
% Squares using PLS-PM approach proposed by Wold (1982).
%
% Input:
% MV is a n x J matrix containing the Manifest Variables
% DP is a J x P boolean matrix defining the block of MVs for each Latent Variable  
% DP is the Outer Design Matrix 
%
% i.e. three LV and six MV (two MVs for each LV)
% DP=[1 0 0
%     1 0 0
%     0 1 0
%     0 1 0
%     0 0 1
%     0 0 1]
%
% DB is a P x P Path Design Matrix
% i.e. Two exogenous LVs connected with an endogenous LV
% Rows represent the exogenous LVs while the column are the endogenous LV
% here the LV1 is the response while LV2 and LV3 are the predictors
%       LV1 LV2 LV3
%   LV1  0   1   1
%   LV2  0   0   0
%   LV3  0   0   0
%
% DB=[0 1 1
%     0 0 0
%     0 0 0]
% 
% DB and DP can be considered as two adjacent matrices.
%
% K is the number of clusters is a priori selected
% Mmode is a string vector indicating, for each block, the used measurement
% model. 'A' for mode A and 'B' for mode B.
% lv_lab is a string vector indicating the label of all MVs
%
% i.e. Considering three LVs (1st and 2nd refrective and 3rd formative)
% Mmode=['A';'A';'B']
%

max_iter=300;
warning('off','all');
n   = size(MV,1);      % number of observations
X   = zscore(MV,1);    % standardization of MVs
Q   = size(DP,2);      % number of LVs
%V   = size(DP,1);     % number of MVs
[~, J]=size(DP);       % J is the number of LVs
eps = 0.0000000000001; % tollerance value

%%%   PLSKM ALGORITHM   %%%    

GoF_ott=0;

Rndstart=25;

for ii=1:Rndstart
    
e   = 1;               % inital value

% Initialization

A=DP;%*rand(J,Q);
A=A*(A'*A)^-0.5; % normalization
U=randPU(n,K);
%U=Ur;
Xmean  = diag(1./sum(U))*U'*X;          %% given U compute Xmean (compute centroids)
Ymean  = Xmean*A;  
Ymean_e = Ymean*A';
%Xp=U*Ymean_e;                          %% reconstructed data matrix
EvolutionA=A;
Evolution_e=e;
DBsquare=DB+triu(DB)';                  %% symmetric Path Design Matrix

iter=0;

while e>eps && iter<max_iter
    iter=iter+1;
    
    % STEP 1: Scores computation

    Xp=U*Xmean;                            %% reconstructed data matrix
    Y=Xp*A;                                %% compute initial scores

    % STEP 2: Inner approximation
    
    W=DBsquare.*corr(Y);                   %% compute inner weights
    Yp=Y*W;                                %% approximate scores
    %tr=trace(Yp'*Yp);
 
    % STEP 3: Outer Approximation
    
    A_old=sum(A,2);
    for j=1:J

        I=find(DP(:,j)>0); % indices of MVs in the j block

        LV=Yp(:,j);
        
        switch Mmode(j)
            case 'A'
                A(I,j)=(1/n).*LV'*Xp(:,I);              %% reflective way
            case 'B'
                A(I,j)=(Xp(:,I)'*Xp(:,I))\Xp(:,I)'*LV;  %% formative way
            otherwise  % default mode
                Mmode(j)='A';
                A(I,j)=(1/n).*Xp(:,I)'*LV;
        end

    end
    
    A=A*(A'*A)^-0.5; % normalization
    A_new=sum(A,2);
    EvolutionA=[EvolutionA A];
    Xmean_old = Xmean;

    % STEP 4: Update U and xmean
    
            U=zeros(n,K);
        for i=1:n
            mindif=sum((X(i,:)-Ymean_e(1,:)).^2);
            posmin=1;
            for j=2:K
                dif=sum((X(i,:)-Ymean_e(j,:)).^2);
                if dif < mindif
                    mindif=dif;
                    posmin=j;
                end 
            end
            U(i,posmin)=1;
        end
        
   % i.e, given U compute Xmean (compute centroids)
   
        su=sum(U);
        while sum(su==0)>0
            [m,p1]=min(su);
            [m,p2]=max(su);
            ind=find(U(:,p2));
            ind=ind(1:floor(su(p2)/2));
            U(ind,p1)=1;
            U(ind,p2)=0;
            su=sum(U);
        end 
            
        %Ymean_e_old=sum(Ymean_e,2);%

        Xmean   = diag(1./sum(U))*U'*X;       %% given U compute Xmean (compute centroids)
        Ymean   = Xmean*A;  
        Ymean_e = Ymean*A';
        
        %Ymean_e_new=sum(Ymean_e,2);%
        
        %eu=sum((Ymean_e_old-Ymean_e_new).^2)  %% Convergence
        e2 = sum(sum((Xmean_old-Xmean).^2)); 
        e  = sum((A_old-A_new).^2);
        Evolution_e=[Evolution_e e];
            
end

%A=A*diag(1./std(X*A,1));  % Weights normalization 
LV=X*A;                    % LV scores
LV=zscore(LV);
%A=corr(X,LV).*DP;         % loadings matrix
%A=A*(A'*A)^-0.5;
[B,R2]=path_coefs(DB,LV);  % path coefficients

% Communality and GoF

loadings=corr(MV,LV).*DP;
com=loadings.^2;
GoF=sqrt(mean(loadings(loadings~=0).^2)*mean(R2(R2~=0)));

devB = trace((U*Ymean_e)'*(U*Ymean_e));
devT = trace(X'*X);

DevRatio=devB/devT;
GoF_adj=sqrt(mean(com(com~=0))*DevRatio*mean(R2(R2~=0)));

if GoF>GoF_ott
    GoF_ott=GoF;
    U_ott=U;
    A_ott=A;
    LV_ott=LV; 
    Y_ott=Y; 
    B_ott=B; 
    R2_ott=R2;
    com_ott=com;
    e_ott=e;
    Evolution_e_ott=Evolution_e;
    iter_ott=iter;
    GoF_adj_ott=GoF_adj;
end

  disp(sprintf('PLS-SEM-KM: nStart=%g,GoF=%g, Adj_GoF=%g, n_Iter=%g, e2=%g',ii, GoF, GoF_adj, iter, e2))

end

  disp(sprintf('PLS-SEM-KM(final): GoF=%g, Adj_GoF=%g, n_Iter=%g, e2=%g', GoF_ott, GoF_adj_ott, iter_ott, e2))

% scores normalization 

for s=1:Q
Y_ott(:,s) = (LV_ott(:,s)-min(LV_ott(:,s)))/(max(LV_ott(:,s))-min(LV_ott(:,s)));
end

% scores histograms
subplot(1,Q,1)
histogram(Y_ott(:,1), 30, 'FaceColor', [0 0.5 0.5])
title(lv_lab(1))
xlim([0 1])
for s=2:Q
subplot(1,Q,s)
histogram(Y_ott(:,s), 30, 'FaceColor', [0 0.5 0.5])
title(lv_lab(s))
xlim([0 1])
end

% scores boxplots
figure
boxplot(Y_ott, {lv_lab(1:Q)})

% groups representation

if K<=8

Ur=zeros(size(MV,1),1);

for k=1:K
for a=1:size(MV,1)
    
if U_ott(a,k)==1
Ur(a,1)=k;
end

end
end

axis square
gplotmatrix(Y_ott,[],Ur,'brkmcywv','+x^d.*os',[],'off','hist',lv_lab);

end

figure
subplot(K,1,1)
boxplot(Y_ott(U_ott(:,1)==1,:),{lv_lab(1:Q)})
title('Group 1')
for k=2:K
subplot(K,1,k)
boxplot(Y_ott(U_ott(:,k)==1,:),{lv_lab(1:Q)})
title(sprintf('Group %g',k))
end

% ***************************%
%                            %
%     RELATED FUNCTIONS      %
%                            %
% ***************************%

% RANDOM U FUNCTION

function [U]=randPU(n,c)

% generates a random partition of n objects in c classes
%
% n = number of objects
% c = number of classes
%
U=zeros(n,c);
U(1:c,:)=eye(c);

U(c+1:n,1)=1;
for i=c+1:n
    U(i,[1:c])=U(i,randperm(c));
end
U(:,:)=U(randperm(n),:);

% PATH COEFFICIENTS FUNCTION

function [B,R2,res]=path_coefs(DB,LV)
% [B,R2,res]=path_coefs(DB,LV)
% This is an internal routine of root function plssem.m. 
% The function calculates inner path coefficents in PLS-PM
%
% INPUT:
%
% DB is the Path Design matrix
%
% LV are the latent Variable scores
%
% OUTPUT:
% 
% B is the matrix of path coefficients
%
% R2 is the vector of R2 coefficient of inner regressions
%
% res is the matrix of inner residuals

[n,q]=size(LV);
endog=find(sum(DB)>0);
JJ=length(endog);
B=DB;
R2=zeros(q,1);
res=zeros(n,q);

for jj=1:JJ
    j=endog(jj);
    k=find(DB(:,j)==1);
    [b,r2,r]=ols(LV(:,j),LV(:,k),1);
    B(k,j)=b(2:end);
    R2(j)=r2;
    res(:,j)=r;
end

% OLS FUNCTION

function [B,R2,res]=ols(Y,X,constant)
%[B R2 res]=ols(Y,X,constant)
%
% Estimate Linear Regression Model through OLS 
%
% INPUT:
% Y is a column vector N x 1 (Response variable)
% X is matrix N x P where each column is a predictor.
% constant is a boolean value equal to: 1 to estimate the model with the intercept and 0 otherwise 
%
% OUTPUT:
% B is the vector of beta coefficients
% R2 is the goodness of fit
% res is the vector of residuals

n=size(X,1);
if constant==1
    X=[ones(n,1) X];
end

XX=inv(X'*X)*X';
H=X*XX;

B=XX*Y;
Yhat=H*Y;
res=Y-Yhat;
DevRes=sum(res.^2);
DevTot=var(Y,1)*n;
R2=1-(DevRes/DevTot);
