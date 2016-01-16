/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <iostream>
#include <fstream>

#include "Features.h"
#include "AnnotationData.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;


// main routine
int main(int argc, char* argv[]) {

  //%%%%%%%%%%%%%%%%%%%%%%%% init %%%%%%%%%%%%%%%%%%%%%%%%

  // read arguments
  if(argc<3) {
    cerr << "Usage: ComputeFeatures config.txt override(no(0)/yes(1))" << endl;
    exit(-1);
  }

  // read config file
  StructParam param;
  if(!param.loadConfigFeature(argv[1])) {
    cerr << "Could not parse " << argv[1] << endl;
    exit(-1);
  }

  // read test/anno data (uses same data structure)
  AnnotationData TestD;
  TestD.loadAnnoFile(param.test_file.c_str());

  //if(atoi(argv[2])==2)
  //system(("rm " + param.feature_path + "/*.pgm").c_str());


  // detect hypotheses on all images
  for(int i=0; i<TestD.AnnoData.size(); ++i) {

    // read image
    Mat originImg = imread((param.image_path+"/"+TestD.AnnoData[i].image_name).c_str());
    if(originImg.empty()) {
      cerr << "Could not read image file " << param.image_path << "/" << TestD.AnnoData[i].image_name << endl;
      continue;
    }

    cout << system(("mkdir " + param.feature_path + "/" + TestD.AnnoData[i].image_name).c_str());

    // extract features
    for(int k=0;k<param.scales.size(); ++k) {

      Features Feat;
      string fname(param.feature_path+"/"+TestD.AnnoData[i].image_name+"/"+TestD.AnnoData[i].image_name);
      if( atoi(argv[2])==1 || !Feat.loadFeatures( fname, param.scales[k]) ) {

	Mat scaledImg;
	resize(originImg, scaledImg, Size(int(originImg.cols * param.scales[k] + 0.5), int(originImg.rows * param.scales[k] + 0.5)) );
	Feat.extractFeatureChannels(scaledImg);
	Feat.saveFeatures( fname, param.scales[k]);

#if 0
	// debug!!!!
	Features Feat2;
	namedWindow( "ShowF", CV_WINDOW_AUTOSIZE );
	imshow( "ShowF", Feat.Channels[0] );

	Feat2.loadFeatures( fname, param.scales[k]);

	namedWindow( "ShowF2", CV_WINDOW_AUTOSIZE );
	imshow( "ShowF2", Feat2.Channels[0] );

	cout << scaledImg.rows << " " << scaledImg.cols << " " << scaledImg.depth() << " " << scaledImg.channels() << " " << scaledImg.isContinuous() << endl;
	cout << Feat.Channels[0].rows << " " << Feat.Channels[0].cols << " " << Feat.Channels[0].depth() << " " << Feat.Channels[0].channels() << " " << Feat.Channels[0].isContinuous() << endl;
	cout << Feat2.Channels[0].rows << " " << Feat2.Channels[0].cols << " " << Feat2.Channels[0].depth() << " " << Feat2.Channels[0].channels() << " " << Feat.Channels[0].isContinuous() << endl;


	Mat diff(Size(scaledImg.cols,scaledImg.rows),CV_8UC1);
	cout << diff.rows << " " << diff.cols << " " << diff.depth() << " " << diff.channels() << " " << scaledImg.isContinuous() << endl;

	diff = Feat.Channels[0] - Feat2.Channels[0];

	namedWindow( "ShowDiff", CV_WINDOW_AUTOSIZE );
	imshow( "ShowDiff", diff );
	waitKey(0);
#endif
      }

    }

  }

  return 0;

}
