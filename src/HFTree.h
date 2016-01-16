/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef HFTreeH
#define HFTreeH

#include "opencv2/core/core.hpp"

#include "Node.h"
#include <iostream>
using namespace std;



class HFTree {

public:

    virtual void checkMemory()=0; // HFTree is an abstract class

protected:

    // pointer to parameters for training
    StructParam* train_param;

    // ratio pos/negative
    int num_pos;
    int num_neg;
    float pov_ratio;
};

class HFTree_L1 : public HFTree {

public:

    ~HFTree_L1();

    //IO functions
    bool readTree(const char* filename, StructParam* par);
    bool saveTree(const char* filename) const;

    // Regression
    LeafNode_L1* regression(uchar** ptFCh, ushort** ptsFCh, int numEntChar, int stepImg);
    LeafNode_L1* regression(PatchFeature_L1* patch);

    // Train
    void train(TrainingData_L1* TrainD, int i, StructParam* param);

    // Debug
    void checkMemory();

private:

    // utility functions
    void saveNode(Node* node, int& nodeId, std::ofstream& out) const;
    void readNode(std::ifstream& in, Node* parent, bool branch);
    Node* readNode(std::ifstream& in, leaf_type& type);

    // Private functions for training
    void grow(InterNode_L1& node, Node* parent);

    // Private training functions
    void makeLeaf(InterNode_L1& node, Node* parent);
    void makeTestNode(InterNode_L1& node, Node* parent, InterNode_L1& left, InterNode_L1& right, BinaryTest_L1& test);

    bool optimizeTest(InterNode_L1& current, InterNode_L1& left, InterNode_L1& right, BinaryTest_L1& test);
    void split(InterNode_L1& current, InterNode_L1& left, InterNode_L1& right, std::vector<IntIndex_L1>& valPos, std::vector<IntIndex_L1>& valNeg, int tao);

    double distMean(InterNode_L1& node);
    double distGain(InterNode_L1& left, InterNode_L1 & right, double prev);
    double entropy(InterNode_L1& node);
    double InfGain(InterNode_L1& left, InterNode_L1& right, double prev);

    double getGainMargin(int mode) {
        if (mode==0) return train_param->inf_margin; else return train_param->dist_margin;
    }
    double measureNode(InterNode_L1& node, int mode) {
        if (mode==0) return entropy(node); else return distMean(node);
    }
    double measureSplit(InterNode_L1& left, InterNode_L1& right, double prev, int mode) {
        if (mode==0) return InfGain(left, right, prev); else return distGain(left,right,prev);
    }

    // Data structure
    std::vector<LeafNode_L1*> leaves;
    std::vector<TestNode_L1*> nodes;

    // pointer to training data
    TrainingData_L1* TrainD;

};

class HFTree_L2 : public HFTree {

public:
    ~HFTree_L2();

    //IO functions
    bool readTree(const char* filename, StructParam* par);
    bool saveTree(const char* filename) const;

    // Regression
    LeafNode_L2* regression(stHist& ipDist);
    LeafNode_L2* regression(std::vector<stHist>& ipDist);

    // Train
    void train(TrainingData_L2* TrainD, int i, StructParam* param);

    // Debug
    void checkMemory();

private:

    // utility functions
    void saveNode(Node* node, std::ofstream& out) const;
    void readNode(std::ifstream& in, Node* parent, bool branch);
    Node* readNode(std::ifstream& in, leaf_type& type);

    // debug function
    void parseTree(Node* node);

    // Private functions for training
    void grow(InterNode_L2& node, Node* parent);

    // Private training functions
    void makeLeaf(InterNode_L2& node, Node* parent);
    void makeTestNode(InterNode_L2& node, Node* parent, InterNode_L2& left, InterNode_L2& right, BinaryTest_L2& test);

    bool optimizeTest(InterNode_L2& current, InterNode_L2& left, InterNode_L2& right, BinaryTest_L2& test);
    void split(InterNode_L2& current, InterNode_L2& left, InterNode_L2& right, std::vector<IntIndex_L2>& valPos, std::vector<IntIndex_L2>& valNeg, float tao);

    double getGainMargin(int mode) {
        if (mode==0) return train_param->inf_margin; else return train_param->dist_margin;
    }
    double measureNode(InterNode_L2& node, int mode) {
        if (mode==0) return entropy(node); else return distMean(node);
    }
    double measureSplit(InterNode_L2& left, InterNode_L2& right, double prev, int mode) {
        if (mode==0) return InfGain(left, right, prev); else return distGain(left,right,prev);
    }
    double distMean(InterNode_L2& node);
    double distGain(InterNode_L2& left, InterNode_L2& right, double prev);
    double entropy(InterNode_L2& node);
    double InfGain(InterNode_L2& left, InterNode_L2& right, double prev);

    // Data structure
    std::vector<LeafNode_L2*> leaves;
    std::vector<TestNode_L2*> nodes;

    // pointer to training data
    TrainingData_L2* TrainD;

};

#endif
