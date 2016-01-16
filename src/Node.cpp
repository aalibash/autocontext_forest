/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "Node.h"

#include <iostream>
#include <fstream>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>


using namespace std;


// make root node
void InterNode_L1::addTrainingData(TrainingData_L1* TrainD, int index) {

    // copy pointer to training patches
    TrainPos.resize( TrainD->new_PosPatches[index].size() );
    for(unsigned int i=0;i<TrainPos.size();++i)
        TrainPos[i] = TrainD->new_PosPatches[index][i];

    TrainNeg.resize( TrainD->new_NegPatches[index].size() );
    for(unsigned int i=0;i<TrainNeg.size();++i)
        TrainNeg[i] = TrainD->new_NegPatches[index][i];

    depth = 0;

}

// release training data
void InterNode_L1::releaseTrainingData(TrainingData_L1* TrainD) {

    //cout << "Release TrainingData IN" << endl;

    // copy pointer to training patches
    for(unsigned int i=0;i<TrainPos.size();++i) {
        TrainD->releaseData(TrainPos[i]);
    }
    for(unsigned int i=0;i<TrainNeg.size();++i) {
        TrainD->releaseData(TrainNeg[i]);
    }

    TrainPos.clear();
    TrainNeg.clear();
    clearInterNode_L1();

}

void InterNode_L1::clearInterNode_L1(){
    vector<BinaryTest_L1>         tempTests;
    vector<vector<int> >       tempTestsThresholds;
    vector<vector<IntIndex_L1> >  tempTestsValPos;
    vector<vector<IntIndex_L1> >  tempTestsValNeg;
    std::vector<PatchFeature_L1*> tempTrainPos;
    std::vector<PatchFeature_L1*> tempTrainNeg;

    Tests.swap(tempTests);
    TestsThresholds.swap(tempTestsThresholds);
    TestsValPos.swap(tempTestsValPos);
    TestsValNeg.swap(tempTestsValNeg);
    TrainPos.swap(tempTrainPos);
    TrainNeg.swap(tempTrainNeg);
}

// debug
void InterNode_L1::checkMemory() {

    // copy pointer to training patches
    for(unsigned int i=0;i<TrainPos.size();++i) {
        cout << TrainPos[i]->id << " ";
    }

}

void InterNode_L1::makeLeaf(LeafNode_L1* node, float pov_ratio) {

    node->pfg = TrainPos.size() / float(pov_ratio*TrainNeg.size()+TrainPos.size());
    node->num_neg = TrainNeg.size();
    node->vCenter.resize( TrainPos.size() );
    for(unsigned int i = 0; i<TrainPos.size(); ++i) {
        node->vCenter[i] = TrainPos[i]->offset;
    }
    node->depth = depth;
    node->branch = branch;

}

void InterNode_L1::copy(InterNode_L1& node) {
    TrainPos = node.TrainPos;
    TrainNeg = node.TrainNeg;
    depth = node.depth;
    branch = node.branch;
}

void LeafNode_L1::save(ofstream& out, int& nodeId) const {

    out << depth << " " << branch << " ";
    out << (int)LEAF_NODE << " " << nodeId << " ";
    out << pfg << " " << num_neg << " ";
    out << vCenter.size() << " ";

    for(unsigned int i=0; i<vCenter.size(); ++i) {
        out << vCenter[i].x << " ";
        out << vCenter[i].y << " ";
    }
    out << endl;

}

void LeafNode_L1::read(ifstream& in) {

    in >> node_id;
    //cout<<"leaf_id : "<<node_id<<endl;
    in >> pfg; in >> num_neg;
    int dummy;
    in >> dummy;
    vCenter.resize(dummy);

    for(unsigned int i=0; i<vCenter.size(); ++i) {
        in >> vCenter[i].x;
        in >> vCenter[i].y;
    }

}

void TestNode_L1::save(ofstream& out, int& nodeId) const {

    out << depth << " " << branch << " " << (int)TEST_NODE << " " << nodeId << " ";
    test.save(out);
    out << endl;

}

void TestNode_L1::read(ifstream& in){

    in>>node_id;
    //cout<<"test_node_id : "<<node_id<<endl;
    test.read(in);

}

/*------------------------------------------------------------------------------------------------
---------------------------   DEFINITIONS FOR LAYER 2 BEGINS   -----------------------------------
------------------------------------------------------------------------------------------------*/

// make root node
void InterNode_L2::addTrainingData(TrainingData_L2* TrainD, int index) {

    // copy pointer to training patches
    TrainPos.resize( TrainD->new_PosPatches[index].size() );
    for(unsigned int i=0;i<TrainPos.size();++i)
    TrainPos[i] = TrainD->new_PosPatches[index][i];

    TrainNeg.resize( TrainD->new_NegPatches[index].size() );
    for(unsigned int i=0;i<TrainNeg.size();++i)
    TrainNeg[i] = TrainD->new_NegPatches[index][i];

    depth = 0;

}

// release training data
void InterNode_L2::releaseTrainingData(TrainingData_L2* TrainD) {

    // copy pointer to training patches
    for(unsigned int i=0;i<TrainPos.size();++i) {
    TrainD->releaseData(TrainPos[i]);
    }
    for(unsigned int i=0;i<TrainNeg.size();++i) {
    TrainD->releaseData(TrainNeg[i]);
    }

    TrainPos.clear();
    TrainNeg.clear();
    clearInterNode();
}

void InterNode_L2::clearInterNode(){
    Tests.clear();
    TestsThresholds.clear();
    TestsValPos.clear();
    TestsValNeg.clear();
    TrainPos.clear();
    TrainNeg.clear();
}

void InterNode_L2::checkMemory() {

    // copy pointer to training patches
    for(unsigned int i=0;i<TrainPos.size();++i) {
        cout << TrainPos[i]->id << " ";
    }
}

void InterNode_L2::makeLeaf(LeafNode_L2* node, float pov_ratio) {

    node->pfg = TrainPos.size() / float(pov_ratio*TrainNeg.size()+TrainPos.size());
    node->num_neg = TrainNeg.size();
    node->vCenter.resize( TrainPos.size() );
    for(unsigned int i = 0; i<TrainPos.size(); ++i) {
        node->vCenter[i].x      = TrainPos[i]->offset.x;
        node->vCenter[i].y      = TrainPos[i]->offset.y;
        node->vCenter[i].width  = TrainPos[i]->size.x;
        node->vCenter[i].height = TrainPos[i]->size.y;
    }
    node->depth = depth;
    node->branch = branch;

}

void InterNode_L2::copy(InterNode_L2& node) {
    TrainPos = node.TrainPos;
    TrainNeg = node.TrainNeg;
    depth = node.depth;
    branch = node.branch;
}

void InterNode_L2::generateIdPool(vector<vector<int> >& thIdPool, int numEntries){
    bool found;
    vector<vector<idPoolElem> > tempIdList;

    // allocate memory
    thIdPool.clear();
    thIdPool.resize(TrainPos[0]->pbDist.size());
    tempIdList.resize(TrainPos[0]->pbDist.size());

    // find freq for all leaf ids
    for(unsigned int treeIdx=0; treeIdx<TrainPos[0]->pbDist.size(); ++treeIdx){
        for(unsigned int grpIdx=0; grpIdx<TrainPos.size(); ++grpIdx){
            for(unsigned int binIdx=0; binIdx<TrainPos[grpIdx]->pbDist[treeIdx].hist.size(); ++binIdx){
                found=0;
                for(unsigned int tempIdListIdx=0; tempIdListIdx<tempIdList[treeIdx].size(); ++tempIdListIdx){
                    if(tempIdList[treeIdx][tempIdListIdx].leafId == TrainPos[grpIdx]->pbDist[treeIdx].hist[binIdx].leafId){
                        tempIdList[treeIdx][tempIdListIdx].freq  += TrainPos[grpIdx]->pbDist[treeIdx].hist[binIdx].freq;
                        found=1;
                        break;
                    }
                }
                if(found == 0){
                    idPoolElem tempIdListElem;
                    tempIdListElem.leafId = TrainPos[grpIdx]->pbDist[treeIdx].hist[binIdx].leafId;
                    tempIdListElem.freq   = TrainPos[grpIdx]->pbDist[treeIdx].hist[binIdx].freq;
                    tempIdList[treeIdx].push_back(tempIdListElem);
                }
            }
        }
        for(unsigned int grpIdx=0; grpIdx<TrainNeg.size(); ++grpIdx){
            for(unsigned int binIdx=0; binIdx<TrainNeg[grpIdx]->pbDist[treeIdx].hist.size(); ++binIdx){
                found=0;
                for(unsigned int tempIdListIdx=0; tempIdListIdx<tempIdList[treeIdx].size(); ++tempIdListIdx){
                    if(tempIdList[treeIdx][tempIdListIdx].leafId == TrainNeg[grpIdx]->pbDist[treeIdx].hist[binIdx].leafId){
                        tempIdList[treeIdx][tempIdListIdx].freq  += TrainNeg[grpIdx]->pbDist[treeIdx].hist[binIdx].freq;
                        found=1;
                        break;
                    }
                }
                if(found == 0){
                    idPoolElem tempIdListElem;
                    tempIdListElem.leafId = TrainNeg[grpIdx]->pbDist[treeIdx].hist[binIdx].leafId;
                    tempIdListElem.freq   = TrainNeg[grpIdx]->pbDist[treeIdx].hist[binIdx].freq;
                    tempIdList[treeIdx].push_back(tempIdListElem);
                }
            }
        }
    }
    for(unsigned int treeIdx=0; treeIdx<TrainPos[0]->pbDist.size(); ++treeIdx){
        sort(tempIdList[treeIdx].begin(),tempIdList[treeIdx].end());
//        cout<<treeIdx<<endl;
//        for(unsigned int idx=tempIdList[treeIdx].size()-25; idx<tempIdList[treeIdx].size(); ++idx){
//            cout<<tempIdList[treeIdx][idx].leafId<<"\t"<<tempIdList[treeIdx][idx].freq<<"\n";
//        }
//        cout<<endl;
    }

    // fill out most frequently occuring entries
    for(unsigned int treeIdx=0; treeIdx<TrainPos[0]->pbDist.size(); ++treeIdx){
        for(int idx=max((int)tempIdList[treeIdx].size()-numEntries,0); idx<(int)tempIdList[treeIdx].size(); ++idx){
            thIdPool[treeIdx].push_back(tempIdList[treeIdx][idx].leafId);
        }
        sort(thIdPool[treeIdx].begin(),thIdPool[treeIdx].end());
//        cout<<treeIdx<<"\t";
//        for(unsigned int idx=0; idx<thIdPool[treeIdx].size(); ++idx){
//            cout<<thIdPool[treeIdx][idx]<<"\t";
//        }
//        cout<<endl;
    }
}

void LeafNode_L2::save(ofstream& out) const {

    out << depth << " " << branch << " ";
    out << (int)LEAF_NODE << " ";
    out << pfg << " " << num_neg << " ";
    out << vCenter.size() << " ";

    for(unsigned int i=0; i<vCenter.size(); ++i) {
        out << vCenter[i].x << " ";
        out << vCenter[i].y << " ";
        out << vCenter[i].width << " ";
        out << vCenter[i].height << " ";
    }
    out << endl;
}

void LeafNode_L2::read(ifstream& in) {

    in >> pfg; in >> num_neg;
    int dummy;
    in >> dummy;
    vCenter.resize(dummy);
    for(unsigned int i=0; i<vCenter.size(); ++i) {
        in >> vCenter[i].x;
        in >> vCenter[i].y;
        in >> vCenter[i].width;
        in >> vCenter[i].height;
    }
}

void TestNode_L2::save(ofstream& out) const {
    out << depth << " " << branch << " " << (int)TEST_NODE << " ";
    test.save(out);
    out << endl;
}

void TestNode_L2::read(std::ifstream& in){
    test.read(in);
}
