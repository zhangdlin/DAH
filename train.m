function [B1, B2, T1, T2] = train(X, Y, param, L)

fprintf('training...\n');

%% set the parameters

r1 = param.r1;
r2 = param.r2;

alphaX = param.alphaX;
alphaY = param.alphaY;
miu = param.miu;
yeta = param.yeta;
beita = param.beita;

lambda = param.lambda;


%% get the dimensions
[n, ~] = size(X);
dY = size(Y,2);

%% transpose the matrices
% X = X'; Y = Y'; L = L';

%% initialization
V1 = randn(n, r1);
V2 = randn(n, r2);


%% Construct the normilized label matrix GTrain
GTrain = NormalizeFea(L,1);
% S = GTrain*GTrain';

%% iterative optimization
for iter = 1:param.iter
     %% update B
     
    B1 = -1*ones(n,r1);
    B1((alphaX*r1*GTrain*(GTrain'*V1)+yeta*V1)>=0) = 1;
    
    B2 = -1*ones(n,r2);
    B2((alphaY*r2*GTrain*(GTrain'*V2)+yeta*V2)>=0) = 1;
    
    
    %% update P
    P1 =  (miu*V1'*V1+lambda*eye(r1)) \ (miu*V1'*L);
    P2 =  (miu*V2'*V2+lambda*eye(r2)) \ (miu*V2'*L);
    
    %% update T
    
    T1 = (beita*V1'*V1+lambda*eye(r1)) \ (beita*V1'*V2);
    T2 = (beita*V1'*V2) / (beita*V2'*V2+lambda*eye(r2));
    
    %% update V
    
    V1 = (alphaX*r1*GTrain*(GTrain'*B1)+beita*(V2*T2'+V2*T1')+yeta*B1+miu*L*P1') / (alphaX*B1'*B1+(beita+yeta)*eye(r1)+beita*T1*T1'+miu*P1*P1'); 
    V2 = (alphaY*r2*GTrain*(GTrain'*B2)+beita*(V1*T2+V1*T1)+yeta*B2+miu*L*P2') / (alphaY*B2'*B2+(beita+yeta)*eye(r2)+beita*T2'*T2+miu*P2*P2');


    

end

clear V1 V2 P1 P2;

end
