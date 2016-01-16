/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <iostream>
#include <fstream>

#include "Param.h"
#include "TrainingData.h"
#include "HFForest.h"

using namespace std;
using namespace cv;

// main routine
int main(int argc, char* argv[]) {

  // read arguments
  if(argc<4) {
    cerr << "Usage: HFTrain config.txt treepath treecounter" << endl;
    exit(-1);
  } 

  // read config file
  Param param; 
  if(!param.loadConfigTrain(argv[1])) {
    cerr << "Could not parse " << argv[1] << endl;
    exit(-1);
  }

  // init forest
  HFForest forest(&param);
  int tree_index = atoi(argv[3]);
    
  // compute seed for random generator
  time_t t = time(NULL);
  int seed = (int)(t/(tree_index+1));
  cout << "Seed " << seed << endl;  

  // read anno file
  TrainingData TrainD;
  TrainD.annoD.loadAnnoFile(param.anno_file.c_str());
  TrainD.initRNG(seed);

  // for each tree to train
  for(int i=0; i<param.ntrees; ++i) {

    // load training data
    TrainD.extractPatches(&param);

    // train tree i
    forest.train(&TrainD, i);

    // save tree i
    forest.saveForest(argv[2], i, tree_index);
    
    // remove training data
    TrainD.clear();

  }

  return 0;

}
