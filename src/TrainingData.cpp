/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "TrainingData.h"

#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



using namespace std;
using namespace cv;

TrainingData_L1::~TrainingData_L1() {
    cout << "Destructor TrainingData_L1 " << PosPatches.size() << " " << NegPatches.size() << endl;

    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
            //cout << i << " ";
            delete PosPatches[i];
        }
    }
    //cout << endl;
    for(unsigned int i=0; i<NegPatches.size();++i) {
        if(NegPatches[i]!=NULL) {
            //cout << i << " ";
            delete NegPatches[i];
        }
    }
    //cout << endl;

    if(rng!=NULL)
    gsl_rng_free(rng);
}

void TrainingData_L1::clear() {

    cout << "Release TrainingData_L1 " << PosPatches.size() << " " << NegPatches.size() << endl;

    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
            //cout << i << " ";
            delete PosPatches[i];
        }
    }
    //cout << endl;
    for(unsigned int i=0; i<NegPatches.size();++i) {
        if(NegPatches[i]!=NULL) {
            //cout << i << " ";
            delete NegPatches[i];
        }
    }
    //cout << endl;

    PosPatches.clear();
    NegPatches.clear();

}

void TrainingData_L1::checkMemory() {

    cout << "Check Memory " << PosPatches.size() << " " << NegPatches.size() << endl;

    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
            cout << i << "/" << PosPatches[i]->id << "/" << PosPatches[i]->label << " ";
        }
    }
    cout << endl << endl;

}

void TrainingData_L1::releaseData(PatchFeature_L1* patch) {
    //cout << "Release Patch " << patch->label << " " << patch->id << endl;
    if(patch->label==-1) {
        //cout << "Free id " << patch->id << " ";
        free_NegPatches.push_back(patch->id);
        NegPatches[patch->id] = NULL;
        delete patch;
    }
    else {
        //cout << "Free id " << patch->id << " " << patch->label << endl;
        free_PosPatches.push_back(patch->id);
        PosPatches[patch->id] = NULL;
        delete patch;
    }
}

void TrainingData_L1::extractPatches(const StructParam* param, int istart, int iend) {

    int i1 = istart; int i2 = iend;
    if(i1<0) i1=0;
    if(i2<0) i2=annoD.AnnoData.size();

    for(int i=i1; i<i2; ++i) {
        // sample with positive sample rate from image set if image contains positive examples
        // sample with negative sample rate from image set if image contains negative examples
        if( ( (annoD.AnnoData[i].posROI.size()>0) &&
        ( (param->samples_images_pos >= 1.0) || (gsl_rng_uniform(rng) < param->samples_images_pos) ) ) ||
        ( (annoD.AnnoData[i].negROI.size()>0) &&
        ( (param->samples_images_neg >= 1.0) || (gsl_rng_uniform(rng) < param->samples_images_neg) ) )
        )
        {
            extractPatches(param, annoD.AnnoData[i]);
        }
    }
}

bool TrainingData_L1::extractPatches(const StructParam* param, Annotation& anno) {

    //cout << "Extract patches from " << anno.image_name << " " << endl;

    // read image
    Mat originImg = imread((param->image_path+"/"+anno.image_name)); // originImg is UINT8_3Channel
    string depthFileName = param->depth_image_path+"/"+anno.image_name.substr(0,anno.image_name.size()-4)+"_abs_smooth.png";
    Mat depthImg  = imread(depthFileName,CV_LOAD_IMAGE_ANYDEPTH);    // depthImg is UINT16

    if(originImg.empty()) {
        cerr << "Could not read image file " << (param->image_path+"/"+anno.image_name) << endl;
        return false;
    }

    // extract and scale positive bounding boxes
    for(unsigned int i=0; i<anno.posROI.size(); ++i) {
        sample_inside(originImg,depthImg,anno.posROI[i],param,0);
    }

    // extract and scale negative bounding boxes
    for(unsigned int i=0; i<anno.negROI.size(); ++i) {
        if(anno.hard_neg){cout<<"hard negative sampling"<<endl; sample_inside(originImg,depthImg,anno.negROI[i],param,-2);}
        else{sample_everywhere(originImg,depthImg,anno.negROI[i],param,-1); cout<<"sampling all over the image for negative examples"<<endl;}
    }

    // extract negative patches outside the positive bounding boxes
    if(anno.sample_background) {
        sample_outside(originImg,depthImg,anno.posROI,anno.negROI.size(),param);
    }

    return true;
}

void TrainingData_L1::sample_inside(Mat& originImg, Mat& depthImg, Rect& bb, const StructParam* param, int label) {

    // get scale factor and offsets
    double scale = (double)param->u_height/(double)bb.height;
    double aspRatio = (double)bb.height / (double)bb.width;
    double dx = 0; //(double)param->p_width/scale;
    double dy = 0; //(double)param->p_height/scale;

    //check if within image
    double a1 = bb.y-dy;
    double a2 = bb.y+bb.height+dy;
    double b1 = bb.x-dx;
    double b2 = bb.x+bb.width+dx;

    // not enough border
    double dy2 = floor(std::min(a1,originImg.rows-a2));
    if(dy2 < 0) {
        dy += dy2;
        a1 = bb.y-dy;
        a2 = bb.y+bb.height+dy;
    }
    double dx2 = floor(std::min(b1,originImg.cols-b2));
    if(dx2 < 0) {
        dx += dx2;
        b1 = bb.x-dx;
        b2 = bb.x+bb.width+dx;
    }

    if(a1>=0 && b1>=0 && a2<=originImg.rows && b2<=originImg.cols) {


        if(depthImg.empty()){

            // extract bounding box
            Mat crop = originImg(Range(a1,a2),Range(b1,b2));
            if(label == 0 || label == -2){
                // scale bounding box
                resize(crop, crop, Size((int)(crop.cols * scale * aspRatio/param->u_asp_ratio + 0.5),(int)(crop.rows * scale + 0.5)), 1.0f, 1.0f,INTER_LANCZOS4);
                cout<<"crop size: "<<crop.rows<<" "<<crop.cols<<endl;
            }
            // extract features
            Features feat;
            feat.extractFeatureChannels(crop);
            // sample
            sample_inside(feat.Channels,param,crop.cols/2,crop.rows/2,label);

        }
        else{
            // extract bounding box
            Mat crop = originImg(Range(a1,a2),Range(b1,b2));
            Mat depthCrop = depthImg(Range(a1,a2),Range(b1,b2));
            if(label == 0 || label == -2){
                // scale bounding box
                resize(crop, crop, Size((int)(crop.cols * scale * aspRatio/param->u_asp_ratio + 0.5),(int)(crop.rows * scale + 0.5)), 1.0f, 1.0f,INTER_LANCZOS4);
                resize(depthCrop, depthCrop, Size((int)(depthCrop.cols*scale*aspRatio/param->u_asp_ratio + 0.5),(int)(depthCrop.rows*scale + 0.5)), 1.0f, 1.0f,INTER_LANCZOS4);
                cout<<"crop size: "<<crop.rows<<" "<<crop.cols<<endl;
            }
            // extract features
            Features feat;
            feat.extractFeatureChannels16(crop,depthCrop);
            // sample
            sample_inside(feat.Channels,param,crop.cols/2,crop.rows/2,label);

        }
    }

    else {

        cerr<<"not sampled from a training instance"<<endl;

        #if 0
        // Debug
        namedWindow( "Show", CV_WINDOW_AUTOSIZE );
        imshow( "Show", originImg );
        waitKey(0);
        #endif

    }

}

void TrainingData_L1::sample_everywhere(Mat& originImg, Mat& depthImg, Rect& bb, const StructParam* param, int label){

    // extract features
    Features feat;
    if(depthImg.empty()){
        feat.extractFeatureChannels(originImg);
    }
    else {
        feat.extractFeatureChannels16(originImg, depthImg);
    }
    // sample
    sample_inside(feat.Channels,param,originImg.cols/2,originImg.rows/2,label);
}

// label -1: negative; 0: positive
void TrainingData_L1::sample_inside(std::vector<cv::Mat>& channels, const StructParam* param, int cx, int cy, int label) {
    int samples;
    if(label==-1){samples = param->samples_bb_neg; cout<<" "<<samples;}
    else if(label == -2){samples = param->samples_bb_neg + 2*param->samples_bg_neg; label = -1; cout<<" "<<samples<<endl;}
    else{samples = param->samples_bb_pos; cout<<" "<<samples;}

    int offx = param->p_width/2;
    int offy = param->p_height/2;
    int x1 = offx;
    int x2 = channels[0].cols-3*offx;
    int y1 = offy;
    int y2 = channels[0].rows-3*offy;

    for(unsigned int index=0; index<new_PosPatches.size();++index) {

        for(int i=0; i<samples; ++i) {

        int x = gsl_rng_uniform_int(rng,x2-x1)+x1;
        int y = gsl_rng_uniform_int(rng,y2-y1)+y1;

        PatchFeature_L1* pf = new PatchFeature_L1();
        pf->Channels.resize(channels.size());

        for(unsigned int c=0;c<channels.size();++c) {

            pf->Channels[c] = channels[c](Range(y,y+param->p_height),Range(x,x+param->p_width));

        }

        pf->label = label;

        if(label==-1) {

            if(free_NegPatches.size()==0) {

                pf->id = NegPatches.size();
                NegPatches.push_back(pf);
                new_NegPatches[index].push_back(pf);

            } else {

                pf->id = free_NegPatches.back();
                NegPatches[pf->id] = pf;
                new_NegPatches[index].push_back(pf);
                free_NegPatches.pop_back();

            }

        }

        else {

            pf->offset.x = x + offx - cx;
            pf->offset.y = y + offy - cy;

            if(free_PosPatches.size()==0) {

                pf->id = PosPatches.size();
                PosPatches.push_back(pf);
                new_PosPatches[index].push_back(pf);
            }

            else {

                pf->id = free_PosPatches.back();
                PosPatches[pf->id] = pf;
                new_PosPatches[index].push_back(pf);
                free_PosPatches.pop_back();

            }

        }


        #if 0
        // Debug
        if(label>=0) {
            namedWindow( "ShowPatch", CV_WINDOW_NORMAL );
            imshow( "ShowPatch", PosPatches.back().Channels[0] );
            waitKey(0);
        }
        #endif

        }
    }
}

void TrainingData_L1::sample_outside(Mat& image, Mat& depthImage, vector<Rect>& bb, int neg, const StructParam* param) {

    // determine number of samples
    int count;
    if(param->samples_bg_neg==-1)
    count = (bb.size()*param->samples_bb_pos - neg*param->samples_bb_neg);
    else
    count = param->samples_bg_neg;

    cout<<" "<<count<<endl;

    // extract features
    Features feat;
    if(depthImage.empty()){
        feat.extractFeatureChannels(image);
    }
    else{
        feat.extractFeatureChannels16(image,depthImage);
    }

    #if 0
    // Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    imshow( "Show", image );
    waitKey(0);
    #endif

    int x1 = 0;
    int x2 = image.cols-param->p_width;
    int y1 = 0;
    int y2 = image.rows-param->p_height;

    for(unsigned int index=0; index<new_PosPatches.size();++index) {

        for(int i=0;i<count;++i) {

        // sampled x,y outside all bounding boxes
        int x,y;
        do {

            x = gsl_rng_uniform_int(rng,x2-x1)+x1;
            y = gsl_rng_uniform_int(rng,y2-y1)+y1;

        } while(is_inside(x,y,bb));

        PatchFeature_L1* pf =  new PatchFeature_L1();
        pf->Channels.resize(feat.Channels.size());

        for(unsigned int c=0;c<feat.Channels.size();++c) {

            pf->Channels[c] = feat.Channels[c](Range(y,y+param->p_height),Range(x,x+param->p_width));

        }
        pf->label = -1;

        if(free_NegPatches.size()==0) {

            pf->id = NegPatches.size();
            NegPatches.push_back(pf);
            new_NegPatches[index].push_back(pf);

        } else {

            pf->id = free_NegPatches.back();
            NegPatches[pf->id] = pf;
            new_NegPatches[index].push_back(pf);
            free_NegPatches.pop_back();

        }

        #if 0
        // Debug
        namedWindow( "ShowPatch", CV_WINDOW_NORMAL );
        imshow( "ShowPatch", NegPatches.back().Channels[0] );
        waitKey(0);
        #endif

        }
    }
}

bool TrainingData_L1::is_inside(int x, int y, std::vector<cv::Rect>& bb) {
    for(unsigned int i=0;i<bb.size();++i) {
        if( is_inside(x,y,bb[i]) ) {
            return true;
        }
    }
    return false;
}

bool TrainingData_L1::is_inside(int x, int y, cv::Rect& bb) {
    return (x>=bb.x && y>=bb.y && x<bb.x+bb.width && y<bb.y+bb.height);
}

/*------------------------------------------------------------------------------------------------
---------------------------   DEFINITIONS FOR LAYER 2 BEGINS   -----------------------------------
------------------------------------------------------------------------------------------------*/

TrainingData_L2::~TrainingData_L2() {

    cout << "Destructor TrainingData_L2 " << PosPatches.size() << " " << NegPatches.size() << endl;
    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
            //cout << i << " ";
            delete PosPatches[i];
        }
    }

    for(unsigned int i=0; i<NegPatches.size();++i) {
        if(NegPatches[i]!=NULL) {
            delete NegPatches[i];
        }
    }

    if(rng!=NULL)
    gsl_rng_free(rng);
}

void TrainingData_L2::clear() {

    cout << "Release TrainingData_L2 " << PosPatches.size() << " " << NegPatches.size() << endl;
    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
        delete PosPatches[i];
        }
    }

    for(unsigned int i=0; i<NegPatches.size();++i) {
        if(NegPatches[i]!=NULL) {
            delete NegPatches[i];
        }
    }

    PosPatches.clear();
    NegPatches.clear();
}

void TrainingData_L2::extractPatches(const StructParam* param, int istart, int iend) {

  int i1 = istart; int i2 = iend;
  if(i1<0) i1=0;
  if(i2<0) i2=annoD.AnnoData.size();
  int hardNegIdx=0;

  for(int i=i1; i<i2; ++i) {

    // sample with positive sample rate from image set if image contains positive examples
    // sample with negative sample rate from image set if image contains negative examples
    if( ( (annoD.AnnoData[i].posROI.size()>0) &&
	  ( (param->samples_images_pos >= 1.0) || (gsl_rng_uniform(rng) < param->samples_images_pos) ) ) ||
	( (annoD.AnnoData[i].negROI.size()>0) &&
	  ( (param->samples_images_neg >= 1.0) || (gsl_rng_uniform(rng) < param->samples_images_neg) ) )
	) {

      extractPatches(param, annoD.AnnoData[i],hardNegIdx);

    }
  }

}

bool TrainingData_L2::extractPatches(const StructParam* param, Annotation& anno, int& hardNegIdx) {

    double scale, aspRatio;
    stringstream treeIdx;
    vector<Rect> tempBb;
    vector<Mat> leafIdMap;
    string fileName = anno.image_name;
    stringstream ssHardNegIdx;

    if(anno.hard_neg==0){
        fileName = param->leafId_path+fileName.substr(0,fileName.size()-4)+"_0_0.txt";
    }
    else{
        ssHardNegIdx.str("");
        ssHardNegIdx<<hardNegIdx;
        hardNegIdx++;
        fileName = param->leafId_path+fileName.substr(0,fileName.size()-4)+"_hardneg_"+ssHardNegIdx.str()+"_0_0.txt";
    }


    readLeafIdImg(fileName,scale,aspRatio,leafIdMap);

//    ofstream out("/home/iai/user/srikanth/Desktop/temp_oldExtr.txt");
//    int* lptr;
//    for(unsigned int y=0; y<leafIdMap[0].rows; ++y){
//        lptr = leafIdMap[0].ptr<int>(y);
//        for(unsigned int x=0; x<leafIdMap[0].cols; ++x, ++lptr){
//            out<<*lptr<<"\t";
//        }
//        out<<endl;
//    }

//    for(unsigned int treeIdx=0; treeIdx<leafIdMap.size(); ++treeIdx){
//        Mat show;
//        leafIdMap[treeIdx].convertTo(show,CV_8U,0.1275);
//        namedWindow("leafIdMap",CV_WINDOW_AUTOSIZE);
//        imshow("leafIdMap",show);
//        waitKey(0);
//    }
//    fileName = fileName.substr(0,fileName.size()-4)+".jpg";
//    Mat originImg = imread(fileName.c_str());
//    namedWindow("originImg",CV_WINDOW_AUTOSIZE);
//    imshow("originImg",originImg);
//    waitKey(0);

    // sample from positive bounding boxes
    tempBb.resize(anno.posROI.size());
    for(unsigned int i=0; i<anno.posROI.size(); ++i){
        tempBb[i].x      = scale*aspRatio*anno.posROI[i].x;
        tempBb[i].y      = scale*anno.posROI[i].y;
        tempBb[i].width  = scale*aspRatio*anno.posROI[i].width;
        tempBb[i].height = scale*anno.posROI[i].height;
//        cout<<tempBb[i].y<<" "<<tempBb[i].x<<" "<<tempBb[i].height<<" "<<tempBb[i].width<<" "<<endl;
    }
    if(anno.posROI.size()>0){sample_inside(leafIdMap,param,tempBb,0);}

    // sample from negative bounding boxes
    tempBb.resize(anno.negROI.size());
    for(unsigned int i=0; i<anno.negROI.size(); ++i){
        tempBb[i].x      = scale*aspRatio*anno.negROI[i].x;
        tempBb[i].y      = scale*anno.negROI[i].y;
        tempBb[i].width  = scale*aspRatio*anno.negROI[i].width;
        tempBb[i].height = scale*anno.negROI[i].height;
    }
    if(anno.negROI.size()>0){
        if(anno.hard_neg){sample_inside(leafIdMap,param,tempBb,-2);}
        else{sample_everywhere(leafIdMap,param,-1);}
    }

    // sample from background
    tempBb.resize(anno.posROI.size());
    for(unsigned int i=0; i<anno.posROI.size(); ++i){
        tempBb[i].x      = scale*aspRatio*anno.posROI[i].x;
        tempBb[i].y      = scale*anno.posROI[i].y;
        tempBb[i].width  = scale*aspRatio*anno.posROI[i].width;
        tempBb[i].height = scale*anno.posROI[i].height;
    }
    if(anno.sample_background){
        sample_outside(leafIdMap,anno.negROI.size(),tempBb,param);
    }

  return true;
}

// read the leaf id file
void TrainingData_L2::readLeafIdImg(const string& fileName, double& scale, double& aspRatio, vector<Mat>& img){

//    int tempNumTrees, tempCol, tempRow, tempInt;
//    ifstream in(fileName.c_str(),std::ios::binary);
//    // read binary file
//    in.read(reinterpret_cast<char*>(&scale),sizeof(scale)); cout<<scale<<endl;
//    in.read(reinterpret_cast<char*>(&aspRatio),sizeof(aspRatio)); cout<<aspRatio<<endl;
//    in.read(reinterpret_cast<char*>(&tempNumTrees),sizeof(tempNumTrees)); cout<<tempNumTrees<<endl;
//    in.read(reinterpret_cast<char*>(&tempRow),sizeof(tempRow)); cout<<tempRow<<endl;
//    in.read(reinterpret_cast<char*>(&tempCol),sizeof(tempCol)); cout<<tempCol<<endl;
//
//    cout<<fileName<<endl<<scale<<" "<<aspRatio<<" "<<tempNumTrees<<" "<<tempRow<<" "<<tempCol<<endl;
//
//    img.resize(tempNumTrees);
//    for(int treeIdx=0; treeIdx<tempNumTrees; ++treeIdx){
//        img[treeIdx].create(tempRow,tempCol,CV_32S);
//    }
//
//    for(int x=0; x<tempCol; ++x){
//        for(int y=0; y<tempRow; ++y){
//            for(int treeIdx=0; treeIdx<tempNumTrees; ++treeIdx){
//                in.read(reinterpret_cast<char*>(&tempInt),sizeof(tempInt));
//                img[treeIdx].at<int>(y,x) = tempInt;
//            }
//        }
//    }
//    in.close();
//    cout<<fileName<<endl<<scale<<" "<<aspRatio<<" "<<tempNumTrees<<" "<<tempRow<<" "<<tempCol<<endl;

    int tempNumTrees, tempCol, tempRow;
    ifstream ipFile(fileName.c_str());

    ipFile >> scale;
    ipFile >> aspRatio;
    ipFile >> tempNumTrees;
    ipFile >> tempRow;
    ipFile >> tempCol;

//    cout<<fileName<<endl<<scale<<" "<<aspRatio<<" "<<tempNumTrees<<" "<<tempRow<<" "<<tempCol<<endl;

    img.resize(tempNumTrees);
    for(int treeIdx=0; treeIdx<tempNumTrees; ++treeIdx){
        img[treeIdx].create(tempRow,tempCol,CV_32S);
    }

    for(int x=0; x<tempCol; ++x){
        for(int y=0; y<tempRow; ++y){
            for(int treeIdx=0; treeIdx<tempNumTrees; ++treeIdx){
                ipFile >> img[treeIdx].at<int>(y,x);
            }
        }
    }
    ipFile.close();
}

bool TrainingData_L2::extractPatches(const StructParam* param, Annotation& anno, vector<vector<Mat> >& leafMappedTrainInstances) {

    double scale, aspRatio;
    stringstream treeIdx;
    vector<Rect> tempBb;

    // sample from positive bounding boxes
    for(unsigned int idx=0; idx<min((unsigned int)1,(unsigned int)anno.posROI.size()) && false==anno.hard_neg; ++idx){

        tempBb.resize(anno.posROI.size());
        scale    = (double)param->u_height/anno.posROI[idx].height;
        aspRatio = (double)anno.posROI[idx].height /anno.posROI[idx].width;
        aspRatio/= param->u_asp_ratio;
        for(unsigned int iterIdx=0; iterIdx<anno.posROI.size(); ++iterIdx){
            tempBb[iterIdx].x      = scale*aspRatio*anno.posROI[iterIdx].x;
            tempBb[iterIdx].y      = scale*anno.posROI[iterIdx].y;
            tempBb[iterIdx].width  = scale*aspRatio*anno.posROI[iterIdx].width;
            tempBb[iterIdx].height = scale*anno.posROI[iterIdx].height;
//            cout<<tempBb[iterIdx].y<<" "<<tempBb[iterIdx].x<<" "<<tempBb[iterIdx].height<<" "<<tempBb[iterIdx].width<<" "<<endl;
        }
        sample_inside(leafMappedTrainInstances[idx],param,tempBb,0);
    }


    // sample from negative bounding boxes
    for(unsigned int idx=0; idx<min((unsigned int)1,(unsigned int)anno.negROI.size()) && true==anno.hard_neg; ++idx){

        tempBb.resize(anno.negROI.size());
        scale    = (double) param->u_height/anno.negROI[idx].height;
        aspRatio = (double) anno.negROI[idx].height / anno.negROI[idx].width;
        aspRatio/= param->u_asp_ratio;
        for(unsigned int iterIdx=0; iterIdx<anno.negROI.size(); ++iterIdx){
            tempBb[iterIdx].x      = scale*aspRatio*anno.negROI[iterIdx].x;
            tempBb[iterIdx].y      = scale*anno.negROI[iterIdx].y;
            tempBb[iterIdx].width  = scale*aspRatio*anno.negROI[iterIdx].width;
            tempBb[iterIdx].height = scale*anno.negROI[iterIdx].height;
        }
        sample_inside(leafMappedTrainInstances[idx],param,tempBb,-2);
    }
    if(anno.negROI.size()>0 && false==anno.hard_neg){

        sample_everywhere(leafMappedTrainInstances[0],param,-1);
    }

    // sample from background
    for(unsigned int idx=0; idx<min((unsigned int)1,(unsigned int)anno.posROI.size()) && true==anno.sample_background; ++idx){

        tempBb.resize(anno.posROI.size());
        scale    = (double)param->u_height/anno.posROI[idx].height;
        aspRatio = (double)anno.posROI[idx].height /anno.posROI[idx].width;
        aspRatio/= param->u_asp_ratio;
        for(unsigned int iterIdx=0; iterIdx<anno.posROI.size(); ++iterIdx){
            tempBb[iterIdx].x      = scale*aspRatio*anno.posROI[iterIdx].x;
            tempBb[iterIdx].y      = scale*anno.posROI[iterIdx].y;
            tempBb[iterIdx].width  = scale*aspRatio*anno.posROI[iterIdx].width;
            tempBb[iterIdx].height = scale*anno.posROI[iterIdx].height;
        }
        sample_outside(leafMappedTrainInstances[idx],anno.negROI.size(),tempBb,param);
    }

  return true;
}

void TrainingData_L2::releaseData(PatchFeature_L2* patch) {

    if(patch->label==-1) {
        free_NegPatches.push_back(patch->id);
        NegPatches[patch->id] = NULL;
        delete patch;
    }
    else {
        free_PosPatches.push_back(patch->id);
        PosPatches[patch->id] = NULL;
        delete patch;
    }
}

void TrainingData_L2::checkMemory() {

    cout << "Check Memory " << PosPatches.size() << " " << NegPatches.size() << endl;
    for(unsigned int i=0; i<PosPatches.size();++i) {
        if(PosPatches[i]!=NULL) {
            cout << i << "/" << PosPatches[i]->id << "/" << PosPatches[i]->label << " ";
        }
    }
    cout << endl << endl;

}

// sample from inside the scale and aspratio normalized bounding box
void TrainingData_L2::sample_inside(vector<Mat>& leafIdMap, const StructParam* param, const vector<Rect>& bb, int label){

    int samples,sIdx=0;
    int *lptr;
    Rect groupbb;
    if(label==-1){
        samples = param->samples_bb_neg;
        //cout<<"neg samples: "<<samples<<endl;
    }
    else if(label == -2){
        samples = param->samples_bb_neg + 2*param->samples_bg_neg;
        label   = -1;
        //cout<<"hard neg samples: "<<samples<<endl;
    }
    else{
        samples = param->samples_bb_pos;
        //cout<<"pos samples: "<<samples<<endl;
    }

//    Mat groupIndicator(leafIdMap[0].rows,leafIdMap[0].cols,CV_8U);
//    groupIndicator.setTo(0);
    for(unsigned int i=0; i<bb.size(); ++i){
//        groupIndicator(Range(bb[i].y,bb[i].y+bb[i].height),Range(bb[i].x,bb[i].x+bb[i].width))=255;
        cout<<bb[i].height<<" "<<bb[i].width<<endl;
    }

    for(unsigned int tree_idx=0; tree_idx<new_PosPatches.size(); ++tree_idx){
        for(unsigned int bb_idx=0; bb_idx<bb.size(); ++bb_idx){
            sIdx=0;
            while(sIdx<samples){
                groupbb.x      = gsl_rng_uniform_int(rng,bb[bb_idx].width)+bb[bb_idx].x;
                groupbb.y      = gsl_rng_uniform_int(rng,bb[bb_idx].height)+bb[bb_idx].y;
                groupbb.x      = max(param->p_width/2,groupbb.x);
                groupbb.y      = max(param->p_height/2,groupbb.y);
                groupbb.width  = param->grp_size_x[gsl_rng_uniform_int(rng,param->grp_size_x.size())];
                groupbb.height = param->grp_size_y[gsl_rng_uniform_int(rng,param->grp_size_y.size())];
                if(is_inside(groupbb,bb[bb_idx]) && groupbb.x+groupbb.width<leafIdMap[0].cols-param->p_width/2 && groupbb.y+groupbb.height<leafIdMap[0].rows-param->p_height/2){
//                    cout<<"groupdetails: "<<groupbb.y<<" "<<groupbb.x<<" "<<groupbb.height<<" "<<groupbb.width<<endl;
                    PatchFeature_L2* grpFeature = new PatchFeature_L2();
                    // get the offset
                    grpFeature->offset.x = bb[bb_idx].x+bb[bb_idx].width/2-groupbb.x;
                    grpFeature->offset.y = bb[bb_idx].y+bb[bb_idx].height/2-groupbb.y;
                    // get the label
                    grpFeature->label = label;
                    // rectangle size
                    grpFeature->size.x = groupbb.width;
                    grpFeature->size.y = groupbb.height;
                    grpFeature->pbDist.resize(leafIdMap.size());
                    for(unsigned int tIdx=0; tIdx<leafIdMap.size(); ++tIdx){
                        // init the histogram
                        grpFeature->pbDist[tIdx].hist.clear();
                        grpFeature->pbDist[tIdx].norm = 0;
                        // compute histogram
                        for(int tempy=groupbb.y; tempy<groupbb.y+groupbb.height; ++tempy){
                            lptr = leafIdMap[tIdx].ptr<int>(tempy)+groupbb.x;
                            for(int tempx=groupbb.x; tempx<groupbb.x+groupbb.width; ++tempx){
                                grpFeature->pbDist[tIdx].update(*lptr);
                                lptr++;
                            }
                        }
                        // sort histogram
                        grpFeature->pbDist[tIdx].sort();
                        grpFeature->pbDist[tIdx].verify();
//                        cout<<" "<<grpFeature->size.y<<" "<<grpFeature->size.x<<endl;
//                        grpFeature->pbDist[tIdx].disp();
                    }
//                    cout<<endl<<endl;
                    // get the id
                    if(label==0){
                        // store the patch
                        grpFeature->id = PosPatches.size();
                        PosPatches.push_back(grpFeature);
                        new_PosPatches[tree_idx].push_back(grpFeature);
                    }
                    else{
                        // store the patch
                        grpFeature->id = NegPatches.size();
                        NegPatches.push_back(grpFeature);
                        new_NegPatches[tree_idx].push_back(grpFeature);
                    }
                    // increment the counter
                    sIdx++;
//                                groupIndicator(Range(groupbb.y,groupbb.y+groupbb.height),Range(groupbb.x,groupbb.x+groupbb.width)) = (unsigned char)(gsl_rng_uniform_int(rng,100)+50);
////                                namedWindow("hello",CV_WINDOW_AUTOSIZE);
////                                imshow("hello",groupIndicator);
////                                waitKey(0);
                }
            }
        }
    }

//    namedWindow("hello",CV_WINDOW_AUTOSIZE);
//    imshow("hello",groupIndicator);
//    waitKey(0);
}

void TrainingData_L2::sample_everywhere(vector<Mat>& leafIdMap, const StructParam* param, int label){
    vector<Rect> fullImgBbVec(1);

    fullImgBbVec[0].x      = 0;
    fullImgBbVec[0].y      = 0;
    fullImgBbVec[0].width  = leafIdMap[0].cols;
    fullImgBbVec[0].height = leafIdMap[0].rows;

    cout<<"sampling everywhere"<<endl;

    sample_inside(leafIdMap, param, fullImgBbVec, label);

}

void TrainingData_L2::sample_outside(vector<Mat>& leafIdMap, const int neg, const vector<Rect>& bb, const StructParam* param){
    int  samples=0;
    int  sIdx=0;
    int  iIdx=0;
    int  maxIter;
    int* lptr;
    Rect groupbb;

    if(param->samples_bg_neg == -1){samples = (bb.size()*param->samples_bb_pos - neg*param->samples_bb_neg);}
    else{samples = param->samples_bg_neg;}
    maxIter=100*samples;

//    Mat groupIndicator(leafIdMap[0].rows,leafIdMap[0].cols,CV_8U);
//    groupIndicator.setTo(0);
//    for(unsigned int i=0; i<bb.size(); ++i){
//        cout<<bb[i].y<<" "<<bb[i].height<<" "<<bb[i].x<<" "<<bb[i].width<<endl;
//        groupIndicator(Range(bb[i].y,bb[i].y+bb[i].height),Range(bb[i].x,bb[i].x+bb[i].width))=255;
//        cout<<bb[i].height<<" "<<bb[i].width<<endl;
//    }

    for(unsigned int tree_idx=0; tree_idx<new_PosPatches.size(); ++tree_idx){
        sIdx=0;
        while(sIdx<samples && iIdx<maxIter){
            iIdx++;
            groupbb.x      = gsl_rng_uniform_int(rng,leafIdMap[0].cols);
            groupbb.y      = gsl_rng_uniform_int(rng,leafIdMap[0].rows);
            groupbb.x      = max(param->p_width/2,groupbb.x);
            groupbb.y      = max(param->p_height/2,groupbb.y);
            groupbb.width  = param->grp_size_x[gsl_rng_uniform_int(rng,param->grp_size_x.size())];
            groupbb.height = param->grp_size_y[gsl_rng_uniform_int(rng,param->grp_size_y.size())];
            if(is_outside(groupbb,bb) && groupbb.x+groupbb.width<(leafIdMap[0].cols-param->p_width/2) && groupbb.y+groupbb.height<(leafIdMap[0].rows-param->p_height/2)){
                //cout<<groupbb.y<<" "<<groupbb.x<<" "<<groupbb.height<<" "<<groupbb.width<<endl;

                PatchFeature_L2* grpFeature = new PatchFeature_L2();
                // get the offset
                grpFeature->offset.x = 0;
                grpFeature->offset.y = 0;
                // get the label
                grpFeature->label  = -1;
                // rectangle size
                grpFeature->size.x = groupbb.width;
                grpFeature->size.y = groupbb.height;
                grpFeature->pbDist.resize(leafIdMap.size());
                for(unsigned int tIdx=0; tIdx<leafIdMap.size(); ++tIdx){
                    // init the histogram
                    grpFeature->pbDist[tIdx].hist.clear();
                    grpFeature->pbDist[tIdx].norm = 0;
                    // compute histogram
                    for(int tempy=groupbb.y; tempy<groupbb.y+groupbb.height; ++tempy){
                        lptr = leafIdMap[tIdx].ptr<int>(tempy)+groupbb.x;
                        for(int tempx=groupbb.x; tempx<groupbb.x+groupbb.width; ++tempx){
                            grpFeature->pbDist[tIdx].update(*lptr);
                            lptr++;
                        }
                    }
                    // sort histogram
                    grpFeature->pbDist[tIdx].sort();
                    grpFeature->pbDist[tIdx].verify();
//                        cout<<" "<<grpFeature->size.y<<" "<<grpFeature->size.x<<endl;
//                        grpFeature->pbDist[tIdx].disp();
                }
                // get the id
                grpFeature->id = NegPatches.size();
                NegPatches.push_back(grpFeature);
                new_NegPatches[tree_idx].push_back(grpFeature);

                // increment the counter
                sIdx++;

//                            groupIndicator(Range(groupbb.y,groupbb.y+groupbb.height),Range(groupbb.x,groupbb.x+groupbb.width)) = (unsigned char)(gsl_rng_uniform_int(rng,100)+50);
//                //            namedWindow("hello",CV_WINDOW_AUTOSIZE);
//                //            imshow("hello",groupIndicator);
//                //            waitKey(0);
            }
        }
    }

    if(iIdx==maxIter){cout<<"could sample only "<<sIdx<<"/"<<samples<<" background groups"<<endl;}

//    namedWindow("hello",CV_WINDOW_AUTOSIZE);
//    imshow("hello",groupIndicator);
//    waitKey(0);

}

bool TrainingData_L2::is_inside(const Rect& tst, const vector<Rect>& refVec){
    for(unsigned int i=0; i<refVec.size(); ++i){
        if(is_inside(tst,refVec[i])){return true;}
    }
    return false;
}

bool TrainingData_L2::is_outside(const Rect& tst, const vector<Rect>& refVec){
    for(unsigned int i=0; i<refVec.size(); ++i){
        if(!is_outside(tst,refVec[i])){return false;}
    }
    return true;
}

bool TrainingData_L2::is_inside(const Rect& tst, const Rect& ref){
    return(tst.x>=ref.x && tst.y>=ref.y && (tst.x+tst.width)<=(ref.x+ref.width) && (tst.y+tst.height)<=(ref.y+ref.height));
}

bool TrainingData_L2::is_outside(const Rect& tst, const Rect& ref){
    return(tst.x>=(ref.x+ref.width) || tst.y>=(ref.y+ref.height) || (tst.x+tst.width)<=ref.x || (tst.y+tst.height)<=ref.y);
}

bool stHist::operator=(const stHist& r) {
    hist.resize(r.hist.size());
    for(unsigned int i=0; i<hist.size(); ++i){
        hist[i].leafId=r.hist[i].leafId;
        hist[i].freq  =r.hist[i].freq;
    }
    norm=r.norm;
    return 1;
}

bool stHist::operator-=(const stHist& rb){
    for(unsigned int bidx=0; bidx<rb.hist.size(); ++bidx){
        for(unsigned int aidx=0; aidx<hist.size(); ++aidx){
            if(rb.hist[bidx].leafId==hist[aidx].leafId){
                hist[aidx].freq-=rb.hist[bidx].freq;
                norm-=rb.hist[bidx].freq;
                break;
            }
        }
    }
    return 1;
}

// adding sorted histograms
bool stHist::operator+=(const stHist& rb){
    stHist tempSum;
    unsigned int aidx=0, bidx=0;
    stHistElem tempHistElem;

    while(aidx<hist.size() && bidx<rb.hist.size()){
        if(hist[aidx].leafId==rb.hist[bidx].leafId){
            tempHistElem.leafId = hist[aidx].leafId;
            tempHistElem.freq   = hist[aidx++].freq+rb.hist[bidx++].freq;
            tempSum.hist.push_back(tempHistElem);
        }else if(hist[aidx].leafId<rb.hist[bidx].leafId){
            tempHistElem.leafId = hist[aidx].leafId;
            tempHistElem.freq   = hist[aidx++].freq;
            tempSum.hist.push_back(tempHistElem);
        }
        else{
            tempHistElem.leafId = rb.hist[bidx].leafId;
            tempHistElem.freq   = rb.hist[bidx++].freq;
            tempSum.hist.push_back(tempHistElem);
        }
    }
    while(aidx<hist.size()){
        tempHistElem.leafId = hist[aidx].leafId;
        tempHistElem.freq   = hist[aidx++].freq;
        tempSum.hist.push_back(tempHistElem);
    }
    while(bidx<rb.hist.size()){
        tempHistElem.leafId = rb.hist[bidx].leafId;
        tempHistElem.freq   = rb.hist[bidx++].freq;
        tempSum.hist.push_back(tempHistElem);
    }
    hist.clear();
    hist.resize(tempSum.hist.size());
    norm=0;
    for(unsigned int i=0; i<hist.size(); ++i){
        hist[i].leafId=tempSum.hist[i].leafId;
        hist[i].freq=tempSum.hist[i].freq;
        norm+=tempSum.hist[i].freq;
    }
    return 1;
}

void stHist::update(int leafId){
    bool found=0;
    for(unsigned int histIdx=0; histIdx<hist.size(); ++histIdx){
        if(leafId==hist[histIdx].leafId){
            hist[histIdx].freq+=1;
            norm+=1;
            found=1;
            break;
        }
    }
    if(found==0 && leafId>=0){
        stHistElem tempHistElem;
        tempHistElem.leafId = leafId;
        tempHistElem.freq = 1;
        hist.push_back(tempHistElem);
        norm+=1;
    }
    if(leafId<0){cout<<"negative leafId seen in Histogram"<<endl;}
}

void stHist::verify(){
    bool bugExists = 0;
    if(hist.size()==0 || norm==0){
        bugExists = 1;
    }
    for(unsigned int i=0; i<hist.size(); ++i){
        if(hist[i].leafId<=0 || hist[i].leafId<=0){
            bugExists=1;
            break;
        }
    }
    for(unsigned int i=1; i<hist.size(); ++i){
        if(hist[i-1].leafId>hist[i].leafId){
            bugExists=1;
            break;
        }
    }
    if(bugExists){
        cout<<"bug in this histogram: "<<endl;
        disp();
    }
}


