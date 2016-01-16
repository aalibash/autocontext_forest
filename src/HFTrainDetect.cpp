/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <iostream>
#include <fstream>
#include <deque>
#include <cmath>
#include <sys/time.h>

#include "Param.h"
#include "TrainingData.h"
#include "HFForest.h"

#include "opencv2/highgui/highgui.hpp"

#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

using namespace std;
using namespace cv;


int train(int argc, char* argv[]) {

    // read arguments
    cout << "Training L1 started" << endl;
    if(argc<4) {
        cerr << "Usage: HFTrainDetect config.txt 0 treecounter" << endl;
        return -1;
    }

    // timer
    timeval start, end;
    double runtime;
    gettimeofday(&start, NULL);

    // read config file
    StructParam param;
    if(!param.loadConfigTrain(argv[1])) {
        cerr << "Could not parse " << argv[1] << endl;
        exit(-1);
    }

//    cout<< "I am here" << endl;

    // init forest
    HFForest_L1 forest(&param);
    int tree_index = atoi(argv[3]);

    // compute seed for random generator
    time_t t = time(NULL);
    int seed = (int)(t/(tree_index+1));
    cout << "Seed " << seed << endl;

    // read anno file
    TrainingData_L1 TrainD;
    TrainD.annoD.loadAnnoFile(param.anno_file.c_str());
    TrainD.annoD.getStatistics(&param);
    TrainD.initRNG(seed);
    TrainD.setCache(param.ntrees_L1);

    cout << "Training Images: " << TrainD.annoD.AnnoData.size() << " Sampling Rate: " << param.samples_images_pos << endl;
    cout << "Patches per pos. BB: " << param.samples_bb_pos << " Patches per neg. BB: " << param.samples_bb_neg << endl;

    // sample patches from training data
    TrainD.extractPatches(&param, 0, TrainD.annoD.AnnoData.size());

    cout << "Sampled patches: " << TrainD.PosPatches.size() << " Sampling Rate: " << TrainD.NegPatches.size() << endl;

    // for each tree to train
    for(int i=0; i<param.ntrees_L1; ++i) {

        // train tree i
        forest.train(&TrainD, i);

        // save tree i
        forest.saveForest((param.treepath_L1).c_str(), i, tree_index);

    }

    // remove training patches from memory
    TrainD.clear();

    gettimeofday(&end, NULL);
    runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
    cout << "Total runtime (L1 train): " << runtime << " msec" << endl;

    return 0;

}

int detect(int argc, char* argv[]) {

    cout << "Testing L1 started" << endl;
    if(argc<3) {
        cerr << "Usage: HFTrainDetect config.txt 1 [image] [detection]" << endl;
        return -1;
    }

    // read config file
    StructParam param;
    if(!param.loadConfigDetect(argv[1])) {
        cerr << "Could not parse " << argv[1] << endl;
        exit(-1);
    }

    // timer
    timeval start, end;
    double runtime;
    gettimeofday(&start, NULL);

    // load forest
    HFForest_L1 forest(&param);
    forest.loadForest(param.treepath_L1);


    AnnotationData TestD;
    TestD.loadAnnoFile(param.test_file.c_str());

    // detect hypotheses on all images
    for(unsigned int i=0; i<TestD.AnnoData.size(); ++i) {

        // read image
        Mat originImg = imread((param.image_path+"/"+TestD.AnnoData[i].image_name).c_str()); // originImg is UINT8_3Channel
        string depthFileName = param.depth_image_path+"/"+TestD.AnnoData[i].image_name.substr(0,TestD.AnnoData[i].image_name.size()-4)+"_abs_smooth.png";
        Mat depthImg  = imread(depthFileName,CV_LOAD_IMAGE_ANYDEPTH);    // depthImg is UINT16

        if(originImg.empty()) {
            cerr << "Could not read image file " << param.image_path << "/" << TestD.AnnoData[i].image_name << endl;
            continue;
        }

        // detect
        Hypotheses hyp;
        if(param.feature_path.empty()) {
            forest.detect(TestD.AnnoData[i].image_name,originImg,depthImg,hyp);
        }

        #if 0
        // evaluate
        Annotation train;
        hyp.check(TestD.AnnoData[i], param.d_thres, train);
        #endif

        // save detections
        hyp.save_detections( (param.hypotheses_path+"/"+TestD.AnnoData[i].image_name+".txt").c_str());

        #if 0
        // show detections
        hyp.show_detections(originImg, param.d_thres);
        #endif

    }

    gettimeofday(&end, NULL);
    runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
    cout << "Total runtime (L1 test): " << runtime << " msec" << endl;

    return 0;

}

int train_L2(int argc, char* argv[]) {

    // read arguments
    cout << "Training L2 started" << endl;
    if(argc<4) {
        cerr << "Usage: HFTrainDetect config.txt 0 treecounter" << endl;
        return -1;
    }

    // read config file
    StructParam param;
    if(!param.loadConfigTrain_L2(argv[1])) {
        cerr << "Could not parse " << argv[1] << endl;
        exit(-1);
    }

    // timer
    timeval start, end;
    gettimeofday(&start, NULL);
    double runtime=0;

    // init first layer forest
    HFForest_L1 forest_L1(&param);
    forest_L1.loadForest(param.treepath_L1);

    // init second layer forest
    HFForest_L2 forest_L2(&param);
    int tree_index = atoi(argv[3]);

    // compute seed for random generator
    time_t t = time(NULL);
    int seed = (int)(t/(tree_index+1));
    cout << "Seed " << seed << endl;

    // read anno file
    TrainingData_L2 TrainD;
    TrainD.annoD.loadAnnoFile(param.anno_file.c_str());
    TrainD.annoD.getStatistics(&param);
    TrainD.initRNG(seed);
    TrainD.setCache(param.ntrees_L2);

    cout << "Training Images: " << TrainD.annoD.AnnoData.size() << " Sampling Rate: " << param.samples_images_pos << endl;
    cout << "Patches per pos. BB: " << param.samples_bb_pos << " Patches per neg. BB: " << param.samples_bb_neg << endl;

    // sample patches from training data
    for(unsigned int sampleIdx=0; sampleIdx<TrainD.annoD.AnnoData.size(); ++sampleIdx){

        // read the image
        string fileName = param.image_path+"/"+TrainD.annoD.AnnoData[sampleIdx].image_name;
        Mat originImg = imread(fileName);
        string depthFileName = param.depth_image_path+"/"+TrainD.annoD.AnnoData[sampleIdx].image_name.substr(0,TrainD.annoD.AnnoData[sampleIdx].image_name.size()-4)+"_abs_smooth.png";
        Mat depthImg  = imread(depthFileName,CV_LOAD_IMAGE_ANYDEPTH);    // depthImg is UINT16

//        namedWindow("show",CV_WINDOW_AUTOSIZE);
//        imshow("show",originImg);
//        waitKey(0);

        // get leaf image from layer 1 forest
        vector<vector<Mat> > leafIdMaps;
        forest_L1.evaluateTrainLeafIdMaps(originImg,depthImg,TrainD.annoD.AnnoData[sampleIdx],leafIdMaps);

        // extract patches
        TrainD.extractPatches(&param, TrainD.annoD.AnnoData[sampleIdx], leafIdMaps);
    }

//    TrainD.extractPatches(&param, 0, TrainD.annoD.AnnoData.size());
    cout << "Sampled patches: " << TrainD.PosPatches.size() << " Sampling Rate: " << TrainD.NegPatches.size() << endl;

    // for each tree to train
    for(int i=0; i<param.ntrees_L2; ++i) {

        // train tree i
        forest_L2.train(&TrainD, i);

        // save tree i
        forest_L2.saveForest((param.treepath_L2).c_str(), i, tree_index);

    }

    // remove training patches from memory
    TrainD.clear();

    gettimeofday(&end, NULL);
    runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
    cout << "Total runtime (L2 train): " << runtime << " msec" << endl;

    return 0;

}

int detect_L2(int argc, char* argv[]) {

    cout << "Testing L2 started" << endl;
    if(argc<3) {
        cerr << "Usage: HFTrainDetect config.txt 1 [image] [detection]" << endl;
        return -1;
    }

    // timer
    timeval start, end;
    gettimeofday(&start, NULL);
    double runtime=0;

    // read config file
    StructParam param;
    if(!param.loadConfigDetect_L2(argv[1])) {
        cerr << "Could not parse " << argv[1] << endl;
        exit(-1);
    }

    // load first layer forest
    HFForest_L1 forest_L1(&param);
    forest_L1.loadForest(param.treepath_L1);

    // load second layer forest
    HFForest_L2 forest_L2(&param);
    forest_L2.loadForest(param.treepath_L2);

    AnnotationData TestD;
    TestD.loadAnnoFile(param.test_file.c_str());

    // detect hypotheses on all images
    for(unsigned int i=0; i<TestD.AnnoData.size(); ++i) {

        // read image
        string fileName = param.image_path+"/"+TestD.AnnoData[i].image_name;
        string depthFileName = param.depth_image_path+"/"+TestD.AnnoData[i].image_name.substr(0,TestD.AnnoData[i].image_name.size()-4)+"_abs_smooth.png";
        Mat originImg = imread(fileName);
        Mat depthImg  = imread(depthFileName,CV_LOAD_IMAGE_ANYDEPTH);    // depthImg is UINT16

        // calculate leaf id maps using first layer forest
        cout<<"evaluating leafId maps"<<endl;
        vector<vector<vector<Mat> > > leafIdMaps;
        forest_L1.evaluateLeafIdMaps(originImg, depthImg, leafIdMaps);

//        // get the vote maps from the first layer
//        cout<<"evaluating L1 vote maps"<<endl;
//        vector<vector<Mat> > voteMaps_L1;
//        forest_L1.returnVoteMaps(originImg, depthImg, voteMaps_L1);

        // get the vote maps from the second layer
        cout<<"evaluating L2 vote maps"<<endl;
        vector<vector<Mat> > voteMaps_L2;
        forest_L2.returnVoteMaps(leafIdMaps,voteMaps_L2,originImg);

#if 0
        namedWindow("show",CV_WINDOW_AUTOSIZE);
        for(unsigned int aspIdx=0; aspIdx<param.asp_ratios.size(); ++aspIdx){
            for(unsigned int sclIdx=0; sclIdx<param.scales.size(); ++sclIdx){
                Mat show;
                voteMaps_L2[aspIdx][sclIdx].convertTo(show,CV_8U,255*0.05);
                imshow("show",show);
                waitKey(0);
            }
        }
#endif

        Hypotheses hyp;
        forest_L2.detect(hyp,voteMaps_L2);
//        hyp.save_detections((param.hypotheses_path+"/"+TestD.AnnoData[i].image_name+".txt").c_str());
//        hyp.show_detections(originImg,param.d_thres);

//        // pass leafIdMaps to second layer forest for detection
//        cout<<"evaluating combined detection"<<endl;
//        vector<Hypotheses> bigHyp;
//        forest_L2.detect(bigHyp, voteMaps_L1, voteMaps_L2);
//
//        // save detections
//        for(unsigned int hypIdx=0; hypIdx<bigHyp.size(); ++hypIdx){
//            char buffer[5];
//            sprintf(buffer,"%02d",hypIdx);
//            string strBuffer = buffer;
//            bigHyp[hypIdx].save_detections( (param.hypotheses_path+"/lambda"+strBuffer+"/"+TestD.AnnoData[i].image_name+".txt").c_str());
//        }
    }

    gettimeofday(&end, NULL);
    runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
    cout << "Total runtime (L2 test): " << runtime << " msec" << endl;

    return 0;

}

// main routine
int main(int argc, char* argv[]) {

    for(int i=0; i<argc; ++i){
        cerr<<argv[i]<<endl;
    }
    cerr<<endl<<endl;

    // read arguments
    if(argc<3) {
        cerr << "Usage: HFTrainDetect config.txt train/test(0/1/2/3) [opt]" << endl;
        exit(-1);
    }

    // train trees or detect images
    if(atoi(argv[2])==0)
        train(argc, argv);
    else if(atoi(argv[2])==1)
        detect(argc, argv);
    else if(atoi(argv[2])==2)
        train_L2(argc,argv);
    else if (atoi(argv[2])==3)
        detect_L2(argc,argv);

    return 0;
}
