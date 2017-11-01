% initialize the LDCNN architecture

function net = ldcnn_initialize(varargin)

opts.batchNormalization = false ; 
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

%**************************************************************************
rng('default');
rng(0) ;

f=1/100 ;   

lr = [1 2] ;
%**************************************************************************

net_pretrained= load(fullfile('model','imagenet-vgg-m')) ;       % the pre-trianed CNN used to build the LDCNN    

net.layers=net_pretrained.layers(1:15);  
net.layers{end+1} = struct('type', 'conv', ...
                           'name','conv6',...
                           'weights', {{f*randn(3,3,512,4096, 'single'), zeros(1,4096,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', [1 1], ...
                           'pad', [0 0 0 0]) ;

net.layers{end+1} = struct('type', 'relu','name','relu6');

net.layers{end+1} = struct('type', 'conv', ...
                           'name','cccp1',...
                           'weights', {{f*randn(1,1,4096,4096, 'single'), zeros(1,4096,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', [1 1], ...
                           'pad', [0 0 0 0]) ;
                       
net.layers{end+1} = struct('type', 'relu','name','relu_cccp1');


net.layers{end+1} = struct('type', 'conv', ...
                           'name','cccp2',...
                           'weights', {{f*randn(1,1,4096,30, 'single'), zeros(1,30,'single')}}, ...
                           'learningRate', 0.001*lr, ...
                           'stride', [1 1], ...
                           'pad', [0 0 0 0]) ;
                       
net.layers{end+1} = struct('type', 'relu','name','relu_cccp2');

net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'avg', ...
                           'pool', [4 4], ...
                           'stride', 1, ...
                           'pad', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;


 drop1 = struct('name', 'dropout6', 'type', 'dropout', 'rate' , 0.5) ;
 drop2 = struct('name', 'dropout7', 'type', 'dropout', 'rate' , 0.5) ;

 net.layers=[net.layers(1:19) drop1 net.layers(20:21) drop2 net.layers(22:end)] ;
                              
%**************************************************************************

% fix the weights of the convolutional layers during training

for i=1:numel(net.layers)-8                                                           
    if isfield(net.layers{i}, 'weights')        
      if ~isfield(net.layers{i}, 'learningRate')                                       
        net.layers{i}.learningRate = [0,0] ;        
      end    
    end
end

%**************************************************************************

% Meta parameters

net.meta.inputSize = [224 224 3] ;   

net.meta.trainOpts.learningRate = [0.001*ones(1,50) 0.0001*ones(1,20) 0.00001*ones(1,30)] ;  

net.meta.trainOpts.batchSize = 4 ;

net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;


% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end
