/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef NodeH
#define NodeH

#include "TrainingData.h"
#include "BinaryTest.h"

// types of nodes
enum leaf_type {LEAF_NODE, TEST_NODE, INTER_NODE};

// Auxilary structure
struct IntIndex_L1 {
    int val;
    int index;
    bool operator<(const IntIndex_L1& a) const { return val<a.val; }
};

struct IntIndex_L2 {
    float val;
    int index;
    bool operator<(const IntIndex_L2& a) const { return val<a.val; }
};

struct Node {

    // Constructor
    Node() {parent=NULL; leftChild=NULL; rightChild=NULL; depth=-1; branch=-1;};
    virtual void save(std::ofstream& out, int& nodeId) const {
        std::cout << "not casted" << std::endl;
    };
    virtual void save(std::ofstream& out) const {
        std::cout << "not casted" << std::endl;
    };
    virtual ~Node() {parent=NULL; leftChild=NULL; rightChild=NULL;};

    int depth;
    // is left child (0) or right child (1) of parent
    bool branch;

    int node_id;

    Node* parent;
    Node* leftChild;
    Node* rightChild;
};

// Structure for L1 test node
struct TestNode_L1 : Node {
    void save(std::ofstream& out, int& nodeId) const;
    void read(std::ifstream& in);
    BinaryTest_L1 test;
};

// Structure for L1 leaves
struct LeafNode_L1 : Node {

    virtual ~LeafNode_L1() { vCenter.clear(); }

    // utilities
    int size() const {return num_neg+vCenter.size();};
    int numPos() const {return vCenter.size();};

    // Probability of foreground
    float pfg;
    // Number of negative
    int num_neg;
    // Vectors from object center to training patches
    std::vector<cv::Point2i> vCenter;

    void save(std::ofstream& out, int& nodeId) const;
    void read(std::ifstream& in);

};

// Structure for L1 intermediate node
struct InterNode_L1 : LeafNode_L1 {

    ~InterNode_L1() { TrainPos.clear(); TrainNeg.clear(); }

    void addTrainingData(TrainingData_L1* TrainD, int index);
    void releaseTrainingData(TrainingData_L1* TrainD);
    void makeLeaf(LeafNode_L1* node, float pov_ratio);
    void copy(InterNode_L1& node);
    void clearInterNode_L1();

    int size() const {return TrainPos.size()+TrainNeg.size();};
    int numPos() const {return TrainPos.size();};

    // debug
    void checkMemory();

    std::vector<PatchFeature_L1*> TrainPos;
    std::vector<PatchFeature_L1*> TrainNeg;

    // pre-store tests and values
    std::vector<BinaryTest_L1> Tests;
    std::vector<std::vector<int> > TestsThresholds;
    std::vector<std::vector<IntIndex_L1> > TestsValPos;
    std::vector<std::vector<IntIndex_L1> > TestsValNeg;
    double current_value;
    int measure_mode;
};

// structure for L2
struct idPoolElem{
    int leafId;
    int freq;
    bool operator<(const idPoolElem& a) const {return freq<a.freq;}
};

// structure for L2 test node
struct TestNode_L2 : Node {
    void save(std::ofstream& out) const;
    void read(std::ifstream& in);
    BinaryTest_L2 test;
};

// structure for L2 leaves
struct LeafNode_L2 : Node {

    virtual ~LeafNode_L2() { vCenter.clear(); }

    void save(std::ofstream& out) const;
    void read(std::ifstream& in);

    //void subsample(int& pos_rem, int& neg_rem, TrainingData* TrainD, double rem_factor);
    int size() const {return num_neg+vCenter.size();};
    int numPos() const {return vCenter.size();};

    // Probability of foreground
    float pfg;
    // Number of negative
    int num_neg;
    // Vectors from object center to training patches + group size information
    std::vector<cv::Rect> vCenter;

};

// Structure for L2 intermediate node
struct InterNode_L2 : LeafNode_L2 {

    ~InterNode_L2() { TrainPos.clear(); TrainNeg.clear(); }

    void addTrainingData(TrainingData_L2* TrainD, int index);
    void releaseTrainingData(TrainingData_L2* TrainD);
    void makeLeaf(LeafNode_L2* node, float pov_ratio);
    void copy(InterNode_L2& node);
    void generateIdPool(std::vector<std::vector<int> >& thIdPool, int numEntries);
    void clearInterNode();

    int size() const {return TrainPos.size()+TrainNeg.size();};
    int numPos() const {return TrainPos.size();};

    // debug
    void checkMemory();

    std::vector<PatchFeature_L2*> TrainPos;
    std::vector<PatchFeature_L2*> TrainNeg;

    // pre-store tests and values
    std::vector<BinaryTest_L2> Tests;
    std::vector<std::vector<float> > TestsThresholds;
    std::vector<std::vector<IntIndex_L2> > TestsValPos;
    std::vector<std::vector<IntIndex_L2> > TestsValNeg;
    double current_value;
    int measure_mode;

};

#endif
