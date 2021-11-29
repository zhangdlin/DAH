% Please note that the demo applies only to Wiki dataset!!!
% If you use other datasets, please select the corresponding parameters!!!

%  If you have any questions, please contact me at anytime(dlinzzhang@gmail.com).
% If you use our code, please cite our article "DAH: Discrete Asymmetric Hashing for Efficient Cross-Media Retrieval".

function Main_testwiki()
clc
clear all;
addpath(genpath('./utils/'));
%% load dataset
 
load('wiki_data.mat');

XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
XTest = I_te; YTest = T_te; LTest = L_te;

clear I_tr I_te T_tr T_te L_tr L_te


%% initialization
fprintf('initializing...\n')

param.r1 = 16;
param.r2 = 64;
param.alphaX = 0.5;
param.alphaY = 0.5;

%% 需要调整的参数
param.miu = 100; 
param.yeta = 100; 
param.beita = 100;
%%

param.lambda = 1e-2;
param.rou = 0.1; 

param.iter = 6;
run = 5;



%% centralization
fprintf('centralizing data...\n');
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

%% kernelization
fprintf('kernelizing...\n\n');
%% Kernel representation
    param.nXanchors = 800; param.nYanchors =1000;
    if 1
        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
        XAnchors = XTrain(anchor_idx,:);
        anchor_idx = randsample(size(YTrain,1), param.nYanchors);
        YAnchors = YTrain(anchor_idx,:);
    else
        [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);
    end
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);
    
    
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end
%% evaluation
 for i = 1 : run   
    %% DAH

    eva_info =evaluate(XKTrain,YKTrain,XKTest,YKTest,LTest,LTrain,param);
    
    % train time
    trainT = eva_info.trainT;
    
    % MAP
    Image_to_Text_MAP = eva_info.Image_to_Text_MAP;
    Text_to_Image_MAP=eva_info.Text_to_Image_MAP;
    map(i,1) =  Image_to_Text_MAP;
    map(i,2) = Text_to_Image_MAP;
    
    fprintf('DAH --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',Image_to_Text_MAP,Text_to_Image_MAP,trainT);

 end
mean(map);
xxx1 = mean(map( : , 1));
xxx2 = mean(map( : , 2));
fprintf('average map over %d runs for Image_to_Text_MAP: %.4f\n', run, xxx1);
fprintf('average map over %d runs for Text_to_Image_MAP: %.4f\n', run, xxx2);


        end
