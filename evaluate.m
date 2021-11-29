function evaluation_info=evaluate(XTrain,YTrain,XTest,YTest,LTest,LTrain,param)
    
    [~,dd1] = size(XTrain);
    [~,dd2] = size(YTrain);
    rou = param.rou;
    r1 = param.r1;
    r2 = param.r2;
    tic;
    [B1, B2, T1, T2] = train(XTrain, YTrain, param, LTrain);
    
    traintime=toc;
    evaluation_info.trainT=traintime;

    D1 = ((XTrain'*XTrain+0.01*eye(dd1)) \ (rou*XTrain'*B1+XTrain'*B2*T1')) / (rou*eye(r1)+T1*T1');
    D2 = ((YTrain'*YTrain+0.01*eye(dd2)) \ (rou*YTrain'*B2+YTrain'*B1*T2))  / (rou*eye(r2)+T2'*T2);
    
    fprintf('evaluating...\n');
    
    %% Training Time
    %traintime=toc;
    %evaluation_info.trainT=traintime;
    topk = 100;
    
    %% image as query to retrieve text database
    BxTest = compactbit((XTest*D1)*T1 > 0);
    ByTrain = compactbit(B2 > 0);
  
    hri2t = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, BxTest, ByTrain, topk);
 
    evaluation_info.Image_to_Text_MAP = hri2t;    

    

    %% text as query to retrieve image database
    ByTest = compactbit((YTest*D2)*T2'> 0);
    BxTrain = compactbit(B1 > 0);
    
    hrt2i = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, ByTest, BxTrain, topk);
    evaluation_info.Text_to_Image_MAP = hrt2i;
 
    
 
                
       
end