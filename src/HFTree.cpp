/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "HFTree.h"
#include <iostream>
#include <fstream>
#include <typeinfo>

using namespace std;


/////////////////////// Constructors /////////////////////////////


HFTree_L1::~HFTree_L1() {

    for(unsigned int i=0;i<nodes.size();++i)
        delete nodes[i];
    for(unsigned int i=0;i<leaves.size();++i) {
        if(typeid(*(leaves[i]))==typeid(InterNode_L1)) {
            delete (static_cast<InterNode_L1*>(leaves[i]));
        }
        else
            delete leaves[i];
    }

    nodes.clear();
    leaves.clear();
}


// Write tree to file
bool HFTree_L1::saveTree(const char* filename) const {
    cout << "Save Tree " << filename << endl;

    bool done = false;
    int nodeId=-1;

    ofstream out(filename);
    if(out.is_open()) {

        out << nodes.size() << " " << leaves.size() << endl;
        out << num_pos << " " << num_neg << endl;
        out << train_param->p_width << " " << train_param->p_height << endl;
        out << train_param->u_width << " " << train_param->u_height << endl;

        saveNode(nodes[0],nodeId,out);

        out.close();

        done = true;
    }

    return done;

};

void HFTree_L1::saveNode(Node* node, int& nodeId, std::ofstream& out) const {
    if(node!=NULL) {
        node->save(out,++nodeId);
        saveNode(node->leftChild,nodeId,out);
        saveNode(node->rightChild,nodeId,out);
    }
}

// Write tree to file
bool HFTree_L1::readTree(const char* filename, StructParam* par) {
    cout << "Read Tree " << filename << endl;

    train_param = par;
    bool done = false;

    ifstream in(filename);
    if(in.is_open()) {

        int dummy;
        in >> dummy;
        cout << "Nodes: " << dummy << " ";
        in >> dummy;
        cout << "Leaves: " << dummy << endl;
        in >> num_pos; in >> num_neg;
        pov_ratio = (float)num_pos / (float)num_neg;

        in >> train_param->p_width;
        in >> train_param->p_height;
        in >> train_param->u_width;
        in >> train_param->u_height;

        readNode(in, NULL, 0);

        in.close();

        cout << endl;

        done = true;
    }

    return done;

}

void HFTree_L1::readNode(std::ifstream& in, Node* parent, bool branch) {

    leaf_type type;
    Node* node = readNode(in, type);
    node->parent = parent;

    switch (type) {

        case LEAF_NODE:
        {

            leaves.push_back( static_cast<LeafNode_L1*>(node) );

            if(parent!=NULL) {
                if(branch) {
                    parent->rightChild = node;
                } else {
                    parent->leftChild = node;
                }
            }
            break;
        }

        case TEST_NODE:
        {
            nodes.push_back( static_cast<TestNode_L1*>(node) );

            if(parent!=NULL) {
                if(branch)
                    parent->rightChild = node;
                else
                    parent->leftChild = node;
            }

            readNode(in,node,0);
            readNode(in,node,1);
            break;
        }

        case INTER_NODE:
        {
            // INTER_NODE not handled
            break;
        }

    }

}

Node* HFTree_L1::readNode(std::ifstream& in, leaf_type& type) {

    int depth; bool branch;
    in >> depth; in >> branch; in >> (int&)type;

    switch(type) {
        case LEAF_NODE:
        {
            LeafNode_L1* node = new LeafNode_L1();
            node->depth = depth;
            node->branch = branch;
            node->read(in);
            return node;
            break;
        }
        case TEST_NODE:
        {
            TestNode_L1* node = new TestNode_L1();
            node->depth = depth;
            node->branch = branch;
            node->read(in);
            return node;
            break;
        }
        case INTER_NODE:
        {
            // INTER_NODE not handled
            break;
        }
    }

    return NULL;
}


LeafNode_L1* HFTree_L1::regression(uchar** ptFCh, ushort** ptsFCh, int numEntChar, int stepImg) {
    Node* pnode = nodes[0];

    while( pnode->leftChild!=NULL ) {
    if( (static_cast<TestNode_L1*>(pnode))->test.evaluate(ptFCh,ptsFCh,numEntChar,stepImg) )
        pnode = pnode->rightChild;
    else
        pnode = pnode->leftChild;
    }

    return (static_cast<LeafNode_L1*>(pnode));
}

LeafNode_L1* HFTree_L1::regression(PatchFeature_L1* patch) {
    Node* pnode = nodes[0];

    while( pnode->leftChild!=NULL ) {
    if( (static_cast<TestNode_L1*>(pnode))->test.evaluate(patch->Channels) )
        pnode = pnode->rightChild;
    else
        pnode = pnode->leftChild;
    }

    return (static_cast<LeafNode_L1*>(pnode));
}

// Start training
void HFTree_L1::train(TrainingData_L1* Train, int index, StructParam* param) {

    train_param = param;
    TrainD = Train;
    num_pos = TrainD->new_PosPatches[index].size();
    num_neg = TrainD->new_NegPatches[index].size();
    pov_ratio = (float)num_pos / (float)num_neg;
    InterNode_L1 root;
    root.addTrainingData(TrainD, index);

    // Grow tree
    grow(root, NULL);

    // Clear cache
    TrainD->clearNew(index);
}

void HFTree_L1::checkMemory() {

    cout << "CheckMemoryTree" << endl;
    for(unsigned int i=0;i<leaves.size();++i) {

        cout << "L" << i << " ";
        if(typeid(*(leaves[i]))==typeid(InterNode_L1)) {
            cout << "IN ";
            (static_cast<InterNode_L1*>(leaves[i]))->checkMemory();
        }

        cout << endl;
    }

    cout << endl;

}

// grow tree and add two children to the node
void HFTree_L1::grow(InterNode_L1& node, Node* parent) {

    //cout << "Grow depth " << node.depth << endl;

    // if depth = max or only many negative
    if(node.depth==train_param->max_depth ) {

        makeLeaf(node, parent);

    }

    else {

        if(node.TrainPos.size()>0 && (node.TrainPos.size()+node.TrainNeg.size()) > (unsigned int)train_param->min_samples_leaf) {

            // Find test and splitting A/B
            BinaryTest_L1 test;
            InterNode_L1 left, right;
            bool valid = optimizeTest(node, left, right, test);

            if(valid) {

                makeTestNode(node,parent,left, right, test);

            }
            else {
                // Could not find valid split
                makeLeaf(node, parent);
            }

        }

        else {
            // Not enough samples
            makeLeaf(node, parent);
        }
    }

}

void HFTree_L1::makeTestNode(InterNode_L1& node, Node* parent, InterNode_L1& left, InterNode_L1& right, BinaryTest_L1& test) {

    // store test
    TestNode_L1* tn = new TestNode_L1();
    tn->test =  test;
    tn->depth = node.depth;
    tn->parent = parent;

    nodes.push_back(tn);

    if(parent!=NULL) {
    if(node.branch)
        parent->rightChild = tn;
    else
        parent->leftChild = tn;
    }

    //cout << "Make test " << node.depth << " " << endl;
    node.clearInterNode_L1();
    left.branch = 0;
    grow(left, tn);
    right.branch = 1;
    grow(right, tn);

}

// Create leaf node
void HFTree_L1::makeLeaf(InterNode_L1& node, Node* parent) {

    LeafNode_L1* leaf = new LeafNode_L1();
    node.makeLeaf(leaf, pov_ratio);

    leaf->parent = parent;

    leaves.push_back(leaf);

    if(parent!=NULL) {
        if(node.branch)
            parent->rightChild = leaf;
        else
            parent->leftChild = leaf;
    }

    //cout << "Make leaf " << leaf->depth << " " << leaf->vCenter.size() << " " << leaf->num_neg << endl;

    node.releaseTrainingData(TrainD);

}

bool HFTree_L1::optimizeTest(InterNode_L1& current, InterNode_L1& left, InterNode_L1& right, BinaryTest_L1& test) {

    double bestGain = 0;
    bool found = false;

    if(current.Tests.size()==0) {

        current.measure_mode = 1;
        if( current.TrainPos.size() < 0.95 * (current.TrainPos.size()+current.TrainNeg.size()) && current.depth < train_param->max_depth-2 )
        current.measure_mode = gsl_rng_uniform_int(TrainD->rng, 2); //TrainD->rng.uniform(0,2);

        current.current_value =  measureNode(current, current.measure_mode);
        double gain_margin = getGainMargin(current.measure_mode);

        current.Tests.resize(train_param->num_test_node);
        current.TestsThresholds.resize(train_param->num_test_node);
        current.TestsValPos.resize(train_param->num_test_node);
        current.TestsValNeg.resize(train_param->num_test_node);

        // generate some tests
        for(int i=0;i<train_param->num_test_node;++i) {

            // generate random test
            current.Tests[i].generate(TrainD->rng, train_param->p_width, train_param->p_height, current.TrainPos[0]->Channels.size());

            // evaluate all patches
            current.TestsValPos[i].resize(current.TrainPos.size());
            for(unsigned int j=0; j<current.TrainPos.size(); ++j) {
                current.TestsValPos[i][j].index = j;
                current.TestsValPos[i][j].val = current.Tests[i].evaluateValue(current.TrainPos[j]->Channels);
            }
            sort( current.TestsValPos[i].begin(), current.TestsValPos[i].end() );

            current.TestsValNeg[i].resize(current.TrainNeg.size());
            for(unsigned int j=0; j<current.TrainNeg.size(); ++j) {
                current.TestsValNeg[i][j].index = j;
                current.TestsValNeg[i][j].val = current.Tests[i].evaluateValue(current.TrainNeg[j]->Channels);
            }
            sort( current.TestsValNeg[i].begin(), current.TestsValNeg[i].end() );

            // get min/max values
            int val_min = current.TestsValPos[i].front().val-1;
            int val_max = current.TestsValPos[i].back().val+2;

            current.TestsThresholds[i].resize(train_param->num_thres_test);

            // generate some test thresholds
            for(int j=0;j<train_param->num_thres_test;++j) {

                // generate tao
                //int tao = TrainD->rng.uniform(val_min, val_max);
                current.TestsThresholds[i][j] = gsl_rng_uniform_int(TrainD->rng, val_max-val_min)+val_min;

                //split data
                InterNode_L1 tmp_left, tmp_right;
                split(current, tmp_left, tmp_right, current.TestsValPos[i], current.TestsValNeg[i], current.TestsThresholds[i][j]);

                // Do not allow small set split (all patches end up in set A or B)
                if( (int)(tmp_left.TrainPos.size()+tmp_left.TrainNeg.size())>train_param->min_samples_leaf
                && (int)(tmp_right.TrainPos.size()+tmp_right.TrainNeg.size())>train_param->min_samples_leaf  ) {

                    // Measure quality of split with measure_mode 0 - classification, 1 - regression
                    double tmpGain = measureSplit(tmp_left, tmp_right, current.current_value, current.measure_mode);

                    // Take binary test with best split
                    if(tmpGain>gain_margin && tmpGain>bestGain) {

                        found = true;
                        bestGain = tmpGain;
                        test = current.Tests[i];
                        test.change_tao(current.TestsThresholds[i][j]);
                        left = tmp_left;
                        left.depth = current.depth + 1;
                        right = tmp_right;
                        right.depth = current.depth + 1;

                    }
                }
            }
        }
    }

    else {
        cout<<"HFTree_L1::OptimizeTest() needs a software patch"<<endl;
        exit(-1);
    }

    return found;

}

void HFTree_L1::split(InterNode_L1& current, InterNode_L1& left, InterNode_L1& right, vector<IntIndex_L1>& valPos, vector<IntIndex_L1>& valNeg, int tao) {

    // search largest value such that val<tao and copy data
    vector<IntIndex_L1>::const_iterator it = valPos.begin();
    while(it!=valPos.end() && it->val<tao) {
        ++it;
    }

    left.TrainPos.resize(it-valPos.begin());
    right.TrainPos.resize(current.TrainPos.size()-left.TrainPos.size());

    it = valPos.begin();
    for(unsigned int i=0; i<left.TrainPos.size(); ++i, ++it) {
        left.TrainPos[i] = current.TrainPos[it->index];
    }

    //it = valPos.begin()+SetA[l].size();
    for(unsigned int i=0; i<right.TrainPos.size(); ++i, ++it) {
        right.TrainPos[i] = current.TrainPos[it->index];
    }

    it = valNeg.begin();
    while(it!=valNeg.end() && it->val<tao) {
        ++it;
    }

    left.TrainNeg.resize(it-valNeg.begin());
    right.TrainNeg.resize(current.TrainNeg.size()-left.TrainNeg.size());

    it = valNeg.begin();
    for(unsigned int i=0; i<left.TrainNeg.size(); ++i, ++it) {
        left.TrainNeg[i] = current.TrainNeg[it->index];
    }

    //it = valPos.begin()+SetA[l].size();
    for(unsigned int i=0; i<right.TrainNeg.size(); ++i, ++it) {
        right.TrainNeg[i] = current.TrainNeg[it->index];
    }

}

double HFTree_L1::distMean(InterNode_L1& node) {

    double mean_x = 0.0;
    double mean_y = 0.0;
    for(unsigned int i=0;i<node.TrainPos.size();++i) {
        mean_x += node.TrainPos[i]->offset.x;
        mean_y += node.TrainPos[i]->offset.y;
    }
    mean_x /= (double)node.TrainPos.size();
    mean_y /= (double)node.TrainPos.size();

    double dist = 0;
    for(unsigned int i=0;i<node.TrainPos.size();++i) {
        double dx = node.TrainPos[i]->offset.x - mean_x;
        double dy = node.TrainPos[i]->offset.y - mean_y;
        dist += (dx*dx+dy*dy);
    }

    return sqrt(dist/double(node.TrainPos.size()));

}

double HFTree_L1::distGain(InterNode_L1& left, InterNode_L1& right, double prev) {

    double p_left =   left.TrainPos.size()+pov_ratio*left.TrainNeg.size();
    double p_right = right.TrainPos.size()+pov_ratio*right.TrainNeg.size();
    p_left = p_left/(p_right+p_left);
    p_right = 1 - p_left;

    return prev - p_left*distMean(left) - p_right*distMean(right);

}

double HFTree_L1::entropy(InterNode_L1& node) {

    double p = double(node.TrainPos.size()) / double(pov_ratio*node.TrainNeg.size()+node.TrainPos.size());
    double entropy = 0;
    if(p>0 && p<1)
    entropy = - p*log(p) - (1-p)*log(1-p);

    return entropy;

}


double HFTree_L1::InfGain(InterNode_L1& left, InterNode_L1& right, double prev) {

    double p_left =   left.TrainPos.size()+pov_ratio*left.TrainNeg.size();
    double p_right = right.TrainPos.size()+pov_ratio*right.TrainNeg.size();
    p_left = p_left/(p_right+p_left);
    p_right = 1 - p_left;

    return prev - p_left*entropy(left) - p_right*entropy(right);

}

/*------------------------------------------------------------------------------------------------
---------------------------   DEFINITIONS FOR LAYER 2 BEGINS   -----------------------------------
------------------------------------------------------------------------------------------------*/

HFTree_L2::~HFTree_L2() {

    for(unsigned int i=0;i<nodes.size();++i)
        delete nodes[i];
    for(unsigned int i=0;i<leaves.size();++i) {
        if(typeid(*(leaves[i]))==typeid(InterNode_L2)) {
            delete (static_cast<InterNode_L2*>(leaves[i]));
        }
        else
            delete leaves[i];
    }

    nodes.clear();
    leaves.clear();
}

// Write tree to file
bool HFTree_L2::saveTree(const char* filename) const {
    cout << "Save Tree " << filename << endl;

    bool done = false;

    ofstream out(filename);
    if(out.is_open()) {

        out << nodes.size() << " " << leaves.size() << endl;
        out << num_pos << " " << num_neg << endl;
        out << train_param->p_width << " " << train_param->p_height << endl;
        out << train_param->u_width << " " << train_param->u_height << endl;

        saveNode(nodes[0], out);

        out.close();

        done = true;
    }

    return done;
};

void HFTree_L2::saveNode(Node* node, std::ofstream& out) const {
    if(node!=NULL) {
        node->save(out);
        saveNode(node->leftChild, out);
        saveNode(node->rightChild, out);
    }
}

void HFTree_L2::parseTree(Node* node) {
    if(node!=NULL) {
        cout << node->depth << " " << node << " ";
        cout << node->leftChild << " " << node->rightChild << endl;

        parseTree(node->leftChild);
        parseTree(node->rightChild);
    }
}

// Write tree to file
bool HFTree_L2::readTree(const char* filename, StructParam* par) {
    cout << "Read Tree " << filename << endl;

    train_param = par;
    bool done = false;

    ifstream in(filename);
    if(in.is_open()) {

        int dummy;
        in >> dummy;
        cout << "Nodes: " << dummy << " ";
        in >> dummy;
        cout << "Leaves: " << dummy << endl;
        in >> num_pos; in >> num_neg;
        pov_ratio = (float)num_pos / (float)num_neg;

        in >> train_param->p_width;
        in >> train_param->p_height;
        in >> train_param->u_width;
        in >> train_param->u_height;

        readNode(in, NULL, 0);

        in.close();

        cout << endl;

        done = true;
    }

    return done;
}

void HFTree_L2::readNode(std::ifstream& in, Node* parent, bool branch) {

    leaf_type type;
    Node* node = readNode(in, type);
    node->parent = parent;

    switch (type) {

        case LEAF_NODE:
        {

            leaves.push_back( static_cast<LeafNode_L2*>(node) );

            if(parent!=NULL) {
                if(branch) {
                    parent->rightChild = node;
                } else {
                    parent->leftChild = node;
                }
            }
            break;
        }

        case TEST_NODE:
        {
            nodes.push_back( static_cast<TestNode_L2*>(node) );

            if(parent!=NULL) {
                if(branch)
                parent->rightChild = node;
                else
                parent->leftChild = node;
            }

            readNode(in,node,0);
            readNode(in,node,1);
            break;
        }

        case INTER_NODE:
        {
        // INTER_NODE not handled
        break;
        }
    }
}

Node* HFTree_L2::readNode(std::ifstream& in, leaf_type& type) {

    int depth; bool branch;
    in >> depth; in >> branch; in >> (int&)type;

    switch(type) {
        case LEAF_NODE:
        {
          LeafNode_L2* node = new LeafNode_L2();
          node->depth = depth;
          node->branch = branch;
          node->read(in);
          return node;
          break;
        }
        case TEST_NODE:
        {
          TestNode_L2* node = new TestNode_L2();
          node->depth = depth;
          node->branch = branch;
          node->read(in);
          return node;
          break;
        }
        case INTER_NODE:
        {
           // INTER_NODE not handled
            break;
        }
    }

    return NULL;
}

LeafNode_L2* HFTree_L2::regression(stHist& ipDist) {

    Node* pnode = nodes[0];
    while( pnode->leftChild!=NULL ) {
        if( (static_cast<TestNode_L2*>(pnode))->test.evaluate(ipDist) )
            pnode = pnode->rightChild;
        else
            pnode = pnode->leftChild;
    }

    return (static_cast<LeafNode_L2*>(pnode));
}

LeafNode_L2* HFTree_L2::regression(vector<stHist>& ipDist) {

    Node* pnode = nodes[0];
    while( pnode->leftChild!=NULL ) {
    if( (static_cast<TestNode_L2*>(pnode))->test.evaluate(ipDist) )
        pnode = pnode->rightChild;
    else
        pnode = pnode->leftChild;
    }

    return (static_cast<LeafNode_L2*>(pnode));
}

// Start training
void HFTree_L2::train(TrainingData_L2* Train, int index, StructParam* param) {

  train_param = param;
  TrainD = Train;
  num_pos = TrainD->new_PosPatches[index].size();
  num_neg = TrainD->new_NegPatches[index].size();

  pov_ratio = (float)num_pos / (float)num_neg;
  InterNode_L2 root;
  root.addTrainingData(TrainD, index);

  // Grow tree
  grow(root, NULL);

  // Clear cache
  TrainD->clearNew(index);

  //cout << "Nodes " << nodes.size() << endl;
  //cout << "Leaves " << leaves.size() << endl;
}

void HFTree_L2::checkMemory() {

    cout << "CheckMemoryTree" << endl;
    for(unsigned int i=0;i<leaves.size();++i) {
        cout << "L" << i << " ";
        if(typeid(*(leaves[i]))==typeid(InterNode_L2)) {
            cout << "IN ";
            (static_cast<InterNode_L2*>(leaves[i]))->checkMemory();
        }
        cout << endl;
    }
    cout << endl;
}

// grow tree and add two children to the node
void HFTree_L2::grow(InterNode_L2& node, Node* parent) {

    //cout << "Grow depth " << node.depth << endl;

    // if depth = max or only many negative
    if(node.depth==train_param->max_depth ) {

        makeLeaf(node, parent);

    }
    else {
        if(node.TrainPos.size()>0 && (node.TrainPos.size()+node.TrainNeg.size()) > (unsigned int)train_param->min_samples_leaf) {
            // Find test and splitting A/B
            BinaryTest_L2 test;
            InterNode_L2 left, right;
            bool valid = optimizeTest(node, left, right, test);

            if(valid) {
                makeTestNode(node,parent,left, right, test);
            }
            else {
                // Could not find valid split
                makeLeaf(node, parent);
            }
        } else {
            // Not enough samples
            makeLeaf(node, parent);
        }
    }
}

void HFTree_L2::makeTestNode(InterNode_L2& node, Node* parent, InterNode_L2& left, InterNode_L2& right, BinaryTest_L2& test) {

    // store test
    TestNode_L2* tn = new TestNode_L2();
    tn->test =  test;
    tn->depth = node.depth;
    tn->parent = parent;

    nodes.push_back(tn);

    if(parent!=NULL) {
        if(node.branch)
            parent->rightChild = tn;
        else
            parent->leftChild = tn;
    }

    //cout << "Make test " << node.depth << " " << endl;
    node.clearInterNode();
    left.branch = 0;
    grow(left, tn);
    right.branch = 1;
    grow(right, tn);

}

void HFTree_L2::makeLeaf(InterNode_L2& node, Node* parent) {

    LeafNode_L2* leaf = new LeafNode_L2();
    node.makeLeaf(leaf, pov_ratio);

    leaf->parent = parent;

    leaves.push_back(leaf);

    if(parent!=NULL) {
        if(node.branch)
            parent->rightChild = leaf;
        else
            parent->leftChild = leaf;
    }

    //cout << "Make leaf " << leaf->depth << " " << leaf->vCenter.size() << " " << leaf->num_neg << endl;

    node.releaseTrainingData(TrainD);
}

bool HFTree_L2::optimizeTest(InterNode_L2& current, InterNode_L2& left, InterNode_L2& right, BinaryTest_L2& test) {

    double bestGain = 0;
    bool found = false;
    int testMode=0; // 0: Jamie Test OR 1: Bhatt Test OR 2:oblique shotton : TODO: export this to config file

    // generate a pool of ids that are most occurring
    vector<vector<int> > thIdPool;
    current.generateIdPool(thIdPool,20);

    if(current.Tests.size()==0) {

        current.measure_mode = 1;
        if( current.TrainPos.size() < 0.95 * (current.TrainPos.size()+current.TrainNeg.size()) && current.depth < train_param->max_depth-2 )
            current.measure_mode = gsl_rng_uniform_int(TrainD->rng, 2); //TrainD->rng.uniform(0,2);

        current.current_value =  measureNode(current, current.measure_mode);
        double gain_margin = getGainMargin(current.measure_mode);

        current.Tests.resize(train_param->num_test_node);
        current.TestsThresholds.resize(train_param->num_test_node);
        current.TestsValPos.resize(train_param->num_test_node);
        current.TestsValNeg.resize(train_param->num_test_node);

//        cout<<endl;

        // generate some tests
        for(int i=0;i<train_param->num_test_node;++i) {

            // generate random test
            if(gsl_rng_uniform_int(TrainD->rng,2) || (current.TrainNeg.size()==0)){

                int pbDistIdx = gsl_rng_uniform_int(TrainD->rng,current.TrainPos.size());
                current.Tests[i].generate(TrainD->rng,current.TrainPos[pbDistIdx]->pbDist,thIdPool,testMode);

            }
            else{

                int pbDistIdx = gsl_rng_uniform_int(TrainD->rng,current.TrainNeg.size());
                current.Tests[i].generate(TrainD->rng,current.TrainNeg[pbDistIdx]->pbDist,thIdPool,testMode);

            }

            // evaluate all positive groups against the test
            current.TestsValPos[i].resize(current.TrainPos.size());
            for(unsigned int j=0; j<current.TrainPos.size(); ++j) {

                current.TestsValPos[i][j].index = j;
                current.TestsValPos[i][j].val = current.Tests[i].evaluateValue(current.TrainPos[j]->pbDist);

            }
            sort(current.TestsValPos[i].begin(), current.TestsValPos[i].end());

            // evaluate all negative groups against the test
            current.TestsValNeg[i].resize(current.TrainNeg.size());
            for(unsigned int j=0; j<current.TrainNeg.size(); ++j) {

                current.TestsValNeg[i][j].index = j;
                current.TestsValNeg[i][j].val = current.Tests[i].evaluateValue(current.TrainNeg[j]->pbDist);

            }
            sort(current.TestsValNeg[i].begin(),current.TestsValNeg[i].end());

            // get min/max values
            float val_min = current.TestsValPos[i].front().val;
            float val_max = current.TestsValPos[i].back().val;

//            if(2==test.returnTreeId() && 632==test.returnLeafId()){
//                cout<<val_min<<"\t"<<val_max<<"\t";
//            }

            // generate some test thresholds
            current.TestsThresholds[i].resize(train_param->num_thres_test);
            for(int j=0;j<train_param->num_thres_test;++j) {

                // generate tao
                current.TestsThresholds[i][j] = (gsl_rng_uniform(TrainD->rng)*(val_max-val_min))+val_min;

                //split data
                InterNode_L2 tmp_left, tmp_right;
                split(current, tmp_left, tmp_right, current.TestsValPos[i], current.TestsValNeg[i], current.TestsThresholds[i][j]);

                // Do not allow small set split (all patches end up in set A or B)
                if( (int)(tmp_left.TrainPos.size()+tmp_left.TrainNeg.size())>train_param->min_samples_leaf
                && (int)(tmp_right.TrainPos.size()+tmp_right.TrainNeg.size())>train_param->min_samples_leaf  ) {

                    // Measure quality of split with measure_mode 0 - classification, 1 - regression
                    double tmpGain = measureSplit(tmp_left, tmp_right, current.current_value, current.measure_mode);

                    // Take binary test with best split
                    if(tmpGain>gain_margin && tmpGain>bestGain) {

                        found = true;
                        bestGain = tmpGain;
                        test = current.Tests[i];
                        test.change_tao(current.TestsThresholds[i][j],testMode);
                        left = tmp_left;
                        left.depth = current.depth + 1;
                        right = tmp_right;
                        right.depth = current.depth + 1;

                    }
                }
            }
        }
    }

    else {

        cerr<<"optimizeTest() functionality not implemented for groups"<<endl;
        exit(-1);

    }

    cout << "OptSplitMeasMode: "<<current.measure_mode<<" OptSplit: " << bestGain<<" "<< current.TrainPos.size()<<" "<<current.TrainNeg.size()<<" "<<left.TrainPos.size()<<" "<<left.TrainNeg.size()<<" "<<right.TrainPos.size()<<" "<<right.TrainNeg.size()<<endl;
    if(found){test.print();} cout<<endl<<"found: "<<found<<endl;

    return found;
}

void HFTree_L2::split(InterNode_L2& current, InterNode_L2& left, InterNode_L2& right, vector<IntIndex_L2>& valPos, vector<IntIndex_L2>& valNeg, float tao) {

    // search largest value such that val<tao and copy data
    vector<IntIndex_L2>::const_iterator it = valPos.begin();
    while(it!=valPos.end() && it->val<tao) {
        ++it;
    }

    left.TrainPos.resize(it-valPos.begin());
    right.TrainPos.resize(current.TrainPos.size()-left.TrainPos.size());

    it = valPos.begin();
    for(unsigned int i=0; i<left.TrainPos.size(); ++i, ++it) {
        left.TrainPos[i] = current.TrainPos[it->index];
    }

    for(unsigned int i=0; i<right.TrainPos.size(); ++i, ++it) {
        right.TrainPos[i] = current.TrainPos[it->index];
    }

    it = valNeg.begin();
    while(it!=valNeg.end() && it->val<tao) {
        ++it;
    }

    left.TrainNeg.resize(it-valNeg.begin());
    right.TrainNeg.resize(current.TrainNeg.size()-left.TrainNeg.size());

    it = valNeg.begin();
    for(unsigned int i=0; i<left.TrainNeg.size(); ++i, ++it) {
        left.TrainNeg[i] = current.TrainNeg[it->index];
    }

    for(unsigned int i=0; i<right.TrainNeg.size(); ++i, ++it) {
        right.TrainNeg[i] = current.TrainNeg[it->index];
    }
}

double HFTree_L2::distMean(InterNode_L2& node) {

    double mean_x = 0.0;
    double mean_y = 0.0;
    for(unsigned int i=0;i<node.TrainPos.size();++i) {
        mean_x += node.TrainPos[i]->offset.x;
        mean_y += node.TrainPos[i]->offset.y;
    }
    mean_x /= (double)node.TrainPos.size();
    mean_y /= (double)node.TrainPos.size();

    double dist = 0;
    for(unsigned int i=0;i<node.TrainPos.size();++i) {
        double dx = node.TrainPos[i]->offset.x - mean_x;
        double dy = node.TrainPos[i]->offset.y - mean_y;
        dist += (dx*dx+dy*dy);
    }

    return sqrt(dist/double(node.TrainPos.size()));

}

double HFTree_L2::distGain(InterNode_L2& left, InterNode_L2& right, double prev) {

    double p_left =   left.TrainPos.size()+pov_ratio*left.TrainNeg.size();
    double p_right = right.TrainPos.size()+pov_ratio*right.TrainNeg.size();
    p_left = p_left/(p_right+p_left);
    p_right = 1 - p_left;

    return prev - p_left*distMean(left) - p_right*distMean(right);

}

double HFTree_L2::entropy(InterNode_L2& node) {

    // entropy: sum_i p_i*log(p_i)
    //double p = double(node.TrainPos.size()) / double(node.TrainNeg.size()+node.TrainPos.size());
    double p = double(node.TrainPos.size()) / double(pov_ratio*node.TrainNeg.size()+node.TrainPos.size());
    double entropy = 0;
    if(p>0 && p<1)
    entropy = - p*log(p) - (1-p)*log(1-p);

    return entropy;

}


double HFTree_L2::InfGain(InterNode_L2& left, InterNode_L2& right, double prev) {

    double p_left =   left.TrainPos.size()+pov_ratio*left.TrainNeg.size();
    double p_right = right.TrainPos.size()+pov_ratio*right.TrainNeg.size();
    p_left = p_left/(p_right+p_left);
    p_right = 1 - p_left;

    return prev - p_left*entropy(left) - p_right*entropy(right);

}

