% construct training and test data

clc; clear;

inPath='./data/AID/Images/';            % data path

tgtDir = dir(inPath);  
nClass = length(tgtDir);

%******************************************%
% the percent of trianing and test images

trainSamplePercent=0.8;                                       
testSamplePercent=0.2;                                               

% the input dimension of LDCNN

nrows=224;                             
ncols=224;
nbands=3;
  
trainSamples=[];                       
trainLabels=[];

testSamples=[];                      
testLabels=[];

%******************************************%

for i = 3:nClass 
    
    curPath = [inPath tgtDir(i).name '/']; 
    
    curDir = dir([curPath, '*.tif']); 
    
    nImgFiles = length(curDir);                          
    index=randperm(nImgFiles);                       
 
    for numTrain=1:nImgFiles*trainSamplePercent;
    
         j = index(numTrain);                                  
         
         trainImg=imread(fullfile(curPath, curDir(j).name));
         
         trainImg=imresize(trainImg,[nrows,ncols]);               
         
         trainSamples=cat(4,trainSamples,trainImg);
         
         sTemp1 = sprintf('%d-%d, %d-%d', i-2, nClass-2, numTrain, nImgFiles*trainSamplePercent); 
         disp('training samples:');
         disp(sTemp1);
         
         if numTrain<=nImgFiles*testSamplePercent
             
             k = index(numTrain+nImgFiles*trainSamplePercent);      
             
             testImg=imread(fullfile(curPath, curDir(k).name));
             
             testImg=imresize(testImg,[nrows,ncols]);               
             
             testSamples=cat(4,testSamples,testImg);
             
             sTemp2 = sprintf('%d-%d, %d-%d', i-2, nClass-2, numTrain, nImgFiles*testSamplePercent); 
             disp('disp test samples:');
             disp(sTemp2);             
             
         end 
     
    end
    
       trainlabel=ones(nImgFiles*trainSamplePercent,1)*(i-2);       
       trainLabels=[trainLabels;trainlabel];                         
       
       testlabel=ones(nImgFiles*testSamplePercent,1)*(i-2);          
       testLabels=[testLabels;testlabel];                            
                     
%******************************************% 

end


disp('trainSamples and testSamples prepared done!!');
disp('trainLabels and testLabels prepared done!!');

save traindata trainSamples trainLabels; 
save testdata testSamples testLabels;         
       
      