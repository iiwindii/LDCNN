function [net, info] = ldcnn_train(varargin)

% -------------------------------------------------------------------------                                                      
%                                                         MatConvNet addpath
% -------------------------------------------------------------------------
run ./matconvnet/matlab/vl_setupnn

% -------------------------------------------------------------------------
%                                                         parameter setting
% -------------------------------------------------------------------------
 opts.dataDir = fullfile('data','AID') ;           % the path of the AID dataset
 
 opts.networkType = 'simplenn' ;
 opts.batchNormalization = false ;                               
 [opts, varargin] = vl_argparse(opts, varargin) ;
 
 sfx = opts.networkType ;
 if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
 opts.expDir = fullfile('data',['AID-' sfx]);                          
 opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');                                           
 opts.train = struct('gpus', 1) ;
 
 [opts, varargin] = vl_argparse(opts, varargin) ;   
 
% -------------------------------------------------------------------------
%                                                    data pre-processing
% -------------------------------------------------------------------------
 opts.ModelMean = true ;                                                       
 opts.contrastNormalization = false ;                        
 opts.whitenData = false;
 opts.trainAugment=false;
 
 opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                            initialize LDCNN architecture
% -------------------------------------------------------------------------
 net = ldcnn_initialize( 'batchNormalization', opts.batchNormalization,...
                                'networkType', opts.networkType);                         
% -------------------------------------------------------------------------
%                                                      prepare the imdb data
% -------------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getAIDImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
                
net.meta.classes.description = imdb.meta.classes.description ;  
net.meta.normalization.averageImage = imdb.images.averageImage ;

% -------------------------------------------------------------------------
%                                                                train CNN
% -------------------------------------------------------------------------
switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end
 
[net, info] = trainfn(net, imdb, getBatch(opts), ...
                         'expDir', opts.expDir, ...
                          net.meta.trainOpts, ...
                          opts.train, ...
                          'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
