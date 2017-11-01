# MatConvNet implementation for low dimensional CNN (LDCNN)
LDCNN is a CNN architecture consisting of common convolutional layers and the mlpconv layer used in Network in Network (NIN). LDCNN 
is proposed in “[Learning Low Dimensional Convolutional Neural Networks for High-Resolution Remote Sensing Image Retrieval](http://www.mdpi.com/2072-4292/9/5/489/htm)”.
# How to use
* Download [MatConvNet](http://www.vlfeat.org/matconvnet/) (1.0-beta23 and 24 are tested) and add it to the root directory. Compile           MatConvNet and call root directory LDCNN.
* Download the [AID](http://www.lmars.whu.edu.cn/xia/AID-project.html) dataset and add it to the data directory, i.e. LDCNN/data/AID
* Download the pretrained [VGGM](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) model and add it to LDCNN/model
* Run `getImgSamples` to randomly select images from AID dataset to construct training and test set and then add them to the AID directory
* Run `ldcnn_train`
# Citation
If you use this work please cite our work:
>@article{zhou2017learning,
  title={Learning Low Dimensional Convolutional Neural Networks for High-Resolution Remote Sensing Image Retrieval},
  author={Zhou, Weixun and Newsam, Shawn and Li, Congmin and Shao, Zhenfeng},
  journal={Remote Sensing},
  volume={9},
  number={5},
  pages={489},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}
