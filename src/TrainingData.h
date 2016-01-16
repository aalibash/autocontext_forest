/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef TrainingDataH
#define TrainingDataH

#include "Param.h"
#include "AnnotationData.h"
#include "Features.h"

#include "gsl/gsl_rng.h"
#include "opencv2/core/core.hpp"



struct PatchFeature {
    cv::Point2i offset;
    int id;
    int label;
};

struct PatchFeature_L1 : PatchFeature {
    std::vector<cv::Mat> Channels;
};

struct stHistElem{
    int leafId;
    float freq;
    bool operator<(const stHistElem& a) const { return leafId<a.leafId; }
};

struct stHist{
    std::vector<stHistElem> hist;
    int norm;

    bool operator=(const stHist& r);
    bool operator+=(const stHist& rb);
    bool operator-=(const stHist& rb);
    void sort(){std::sort(hist.begin(),hist.end());}
    void disp(){
        std::cout<<"histogram: "<<norm<<std::endl;
        for(unsigned int i=0; i<hist.size(); ++i){std::cout<<hist[i].leafId<<" "<<hist[i].freq<<"  ";}
        std::cout<<std::endl;
    }
    void update(int leafId);
    void verify();
};

struct PatchFeature_L2 : PatchFeature {
    cv::Point2i size;
    std::vector<stHist> pbDist;
};

class TrainingData {

public:

    virtual void clear()=0; // Training data is an abstract class

    // init random generator
    void initRNG(int seed) {
        rng = gsl_rng_alloc(gsl_rng_taus);
        gsl_rng_set(rng, seed);
    }

    // random generator
    gsl_rng* rng;

    // annotation data
    AnnotationData annoD;

};

class TrainingData_L1 : public TrainingData {

public:
    ~TrainingData_L1();

    // set the number of caches (one cache per tree)
    void setCache(int t) {
        new_PosPatches.resize(t);
        new_NegPatches.resize(t);
    }

    // remove training data
    void clear();

    // label training data as not new
    void clearNew(int i) {
        new_PosPatches[i].clear();
        new_NegPatches[i].clear();
    }

    // extract patches
    void extractPatches(const StructParam* param, int istart = -1, int iend = -1);
    bool extractPatches(const StructParam* param, Annotation& anno);
    void releaseData(PatchFeature_L1* patch);

    // debug
    void checkMemory();

    // data
    std::vector<PatchFeature_L1*> PosPatches;
    std::vector<PatchFeature_L1*> NegPatches;

    // cache for trees
    std::vector<std::vector<PatchFeature_L1*> > new_PosPatches;
    std::vector<std::vector<PatchFeature_L1*> > new_NegPatches;

    // free slots
    std::vector<int> free_PosPatches;
    std::vector<int> free_NegPatches;

private:

    // sample functions for image patches
    // label -1: negative; 0: positive
    void sample_inside(cv::Mat& image, cv::Mat& depthImg, cv::Rect& bb, const StructParam* param, int label);
    void sample_inside(std::vector<cv::Mat>& channels, const StructParam* param, int cx, int cy, int label);
    void sample_everywhere(cv::Mat& image, cv::Mat& depthImg, cv::Rect& bb, const StructParam* param, int label);
    void sample_outside(cv::Mat& image, cv::Mat& depthImg, std::vector<cv::Rect>& bb, int neg, const StructParam* param);
    bool is_inside(int x, int y, std::vector<cv::Rect>& bb);
    bool is_inside(int x, int y, cv::Rect& bb);


};

class TrainingData_L2 : public TrainingData {

public:
    ~TrainingData_L2();

    // set the number of caches (one cache per tree)
    void setCache(int t) {
        new_PosPatches.resize(t);
        new_NegPatches.resize(t);
    }

    // remove training data
    void clear();

    // label training data as not new
    void clearNew(int i) {
        new_PosPatches[i].clear();
        new_NegPatches[i].clear();
    }

    // extract patches
    bool extractPatches(const StructParam* param, Annotation& anno, std::vector<std::vector<cv::Mat> >& leafMappedTrainInstances);
    void extractPatches(const StructParam* param, int istart, int iend);
    bool extractPatches(const StructParam* param, Annotation& anno, int& hardNegIdx);
    void readLeafIdImg(const std::string& fileName, double& scale, double& aspRatio, std::vector<cv::Mat>& img);

    void releaseData(PatchFeature_L2* group);

    // debug
    void checkMemory();

    // data
    std::vector<PatchFeature_L2*> PosPatches;
    std::vector<PatchFeature_L2*> NegPatches;

    // cache for trees
    std::vector<std::vector<PatchFeature_L2*> > new_PosPatches;
    std::vector<std::vector<PatchFeature_L2*> > new_NegPatches;

    // free slots
    std::vector<int> free_PosPatches;
    std::vector<int> free_NegPatches;

private:

    // sample functions for image patches
    // label -1: negative; 0: positive
    void sample_inside(std::vector<cv::Mat>& leafIdMap, const StructParam* param, const std::vector<cv::Rect>& bb, int label);
    void sample_everywhere(std::vector<cv::Mat>& leafIdMap, const StructParam* param, int label);
    bool is_inside(const cv::Rect& tst, const std::vector<cv::Rect>& refVec);
    bool is_inside(const cv::Rect& tst, const cv::Rect& ref);
    bool is_outside(const cv::Rect& tst, const std::vector<cv::Rect>& refVec);
    bool is_outside(const cv::Rect& tst, const cv::Rect& ref);
    void sample_outside(std::vector<cv::Mat>& leafIdMap, const int neg, const std::vector<cv::Rect>& bb, const StructParam* param);
};



#endif
