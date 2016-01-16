/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef HFForestH
#define HFForestH

#include "HFTree.h"
#include "Hypotheses.h"
#include "FPFNmodel.h"

class HFForest {

public:

    virtual void loadForest(std::string filename)=0; // HFForest is an abstract class

protected:

    // Detect hypotheses in Hough space
    void detect(Hypotheses& hyp, std::vector<std::vector<cv::Mat> >& bigVote);

    // Voting space
    std::vector<std::vector<cv::Mat> > bigVote;

    // Voting Scores
    std::vector<float> PosScores;
    std::vector<float> NegScores;

    // Pointer to parameters
    StructParam* param;

};

class HFForest_L1 : public HFForest {

public:

    // Constructors
    HFForest_L1(StructParam* par) {
        param = par;
        vTrees.resize(par->ntrees_L1);
    }

    // IO functions
    void loadForest(std::string filename);
    void saveForest(const char* filename, int index, int index_offset);
    void saveForest(const char* filename, int index_offset = 0);

    // for interfacing with L2
    void returnVoteMaps(cv::Mat& originImg, cv::Mat& depthImg, std::vector<std::vector<cv::Mat> > &voteMaps);
    void evaluateLeafIdMaps(cv::Mat& originImg, cv::Mat& depthImg, std::vector<std::vector<std::vector<cv::Mat> > >& leafIdMaps);
    void evaluateTrainLeafIdMaps(cv::Mat& originImg, cv::Mat& depthImg, Annotation& anno, std::vector<std::vector<cv::Mat> >& leafMappedTrainInstances);
    void getIdMaps(Features& feat, std::vector<cv::Mat>& idMap);

    // Training
    void train(TrainingData_L1* TrainD, int index) {
        vTrees[index].train(TrainD, index, param);
    }

    // Detection
    using HFForest::detect;
    void detect(std::string& fileName, cv::Mat& originImg, cv::Mat& depthImg, Hypotheses& hyp, std::string feature_name = "");

private:

    // Regression
    void regression(std::vector<LeafNode_L1*>& result, uchar** ptFCh, ushort** ptsFCh, int numEntChar, int stepImg);
    // Voting
    void vote(cv::Mat& Vote, Features& Feat);
    // Trees
    std::vector<HFTree_L1> vTrees;

};

class HFForest_L2 : public HFForest {


public:

    // Constructors
    HFForest_L2(StructParam* par) {
        param = par;
        vTrees.resize(par->ntrees_L2);
    }

    // IO functions
    void loadForest(std::string filename);
    void saveForest(const char* filename, int index_offset = 0);
    void saveForest(const char* filename, int index, int index_offset);

    // Training
    void train(TrainingData_L2* TrainD, int index) {
        vTrees[index].train(TrainD, index, param);
    }

    // Detection
    using HFForest::detect;
    void detect(vector<vector<vector<cv::Mat> > >leafIdMap, cv::Mat& originImg, Hypotheses& hyp);
    void detect(std::vector<cv::Mat>& leafIdMap, cv::Mat& voteMap);
    void detect(std::vector<Hypotheses>& bigHyp, std::vector<std::vector<cv::Mat> >& bigVote_L1, std::vector<std::vector<cv::Mat> >& bigVote_L2);

    void returnVoteMaps(std::vector<std::vector<std::vector<cv::Mat> > > leafIdMaps, std::vector<std::vector<cv::Mat> >& voteMaps, cv::Mat& originImg);

private:

    // Regression
    void regression(std::vector<LeafNode_L2*>& result, stHist& ipDist);
    void regression(std::vector<LeafNode_L2*>& result, std::vector<stHist>& ipDist);
    void vote(std::vector<LeafNode_L2*> result, const int& y, const int& x, const int& gw, const int& gh, cv::Mat& voteMap);

    // Trees
    std::vector<HFTree_L2> vTrees;

};




#endif
