/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/


#include "Param.h"
#include "Features.h"
#include "HFForest.h"

#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// main routine
int main(int argc, char* argv[]) {

  // read arguments
  if(argc<4) {
    cerr << "Usage: HFDetect config.txt image detection.txt" << endl;
    exit(-1);
  } 

  // read config file
  Param param; 
  if(!param.loadConfigDetect(argv[1])) {
    cerr << "Could not parse " << argv[1] << endl;
    exit(-1);
  }
    
  // load forest
  HFForest forest(&param);
  forest.loadForest(param.treepath);
  //forest.saveForest(param.treepath.c_str(), 100);
  
  // read image
  Mat originImg = imread(argv[2]);
  if(originImg.empty()) {
    cerr << "Could not read image file " << argv[2] << endl;
    exit(-1);
  }

  // detect
  Hypotheses hyp;
  forest.detect(originImg, hyp);

  // save detections
  hyp.save_detections(argv[3]);

#if 1
  // show detections
  hyp.show_detections(originImg, param.d_thres);
#endif

  return 0;

}
