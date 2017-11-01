% construct the imdb structure
% --------------------------------------------------------------------
function imdb = getAIDImdb(opts)
% --------------------------------------------------------------------

if ~exist(opts.dataDir, 'dir')
    
  mkdir(opts.dataDir) ;
  
end

trainImg=load(fullfile(opts.dataDir, 'traindata'));             


if opts.trainAugment
    
    x_train=trainImg.trainSamples;
    x_train=single(x_train);                                   
    
    x_train_d90=rot90(x_train);                                
    x_train_d180=rot90(x_train,2);
    x_train_d270=rot90(x_train,3);
    x_train_flip=fliplr(x_train);                             
    
    x_train=cat(4, x_train_d90, x_train_d180, x_train_d270, x_train_flip); 
    
    y_train=trainImg.trainLabels;                            
    
    y_train_d90=y_train;
    y_train_d180=y_train;
    y_train_d270=y_train;
    y_train_flip=y_train;
    
    y_train=cat(1,y_train_d90, y_train_d180, y_train_d270, y_train_flip);
    y_train=reshape(y_train,1,[]);

else
    
    x_train=trainImg.trainSamples;
    x_train=single(x_train);                                   
    y_train=trainImg.trainLabels;
    y_train=reshape(y_train,1,[]);
    
end

testImg=load(fullfile(opts.dataDir, 'testdata'));         

x_test=testImg.testSamples;
x_test=single(x_test);                                     
y_test=testImg.testLabels;
y_test=reshape(y_test,1,[]);

%*******************************************************************************

set = [ones(1,numel(y_train)) 3*ones(1,numel(y_test))];   
                                                             
                                                              
inputData = cat(4, x_train, x_test);                        

%*******************************************************************************

if opts. ModelMean
    
   net=load(fullfile('model','imagenet-vgg-m')) ;      
   
   dataMean = single(net.meta.normalization.averageImage);         
    
else
    
   dataMean = mean(inputData(:,:,:,set == 1), 4);            
   
end

inputData = bsxfun(@minus, inputData, dataMean) ;

%********************************************************************************

imdb.images.data = inputData ;
imdb.images.averageImage = dataMean;
imdb.images.labels = cat(2, y_train, y_test) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;                                         
imdb.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:30,'uniformoutput',false) ;  

class_description={'Airport','Bareland','BaseballField','Beach','Bridge','Center','Church'...
    'Commercial','DenseResidential','Desert','Farmland','Forest','Industrial','Meadow',...
    'MediumResidential','Mountain','Park','Parking','Playground','Pond','Port','RailwayStation',...
    'Resort','River','School','SparseResidential','Square','Stadium',...
    'StorageTanks','Viaduct'};
imdb.meta.classes.description=class_description;   
