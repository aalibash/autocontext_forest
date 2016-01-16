/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "Hypotheses.h"

#include <iostream>
#include <fstream>

#include "gsl/gsl_sf_gamma.h"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// evaluate detections
int Hypotheses::check(Annotation& anno, double thres, Annotation& train) {

    train.posROI.clear();
    train.negROI.clear();

    threshold = thres;
    int count_errors = 0;

    vector<Rect> gt = anno.posROI;

    //cout << "Evaluate " << endl;

    int k=0;
    int end_k = detections.size();
    for(; k<end_k; ++k) {

        if(detections[k].conf < thres)
        break;

        int index = detections[k].check(gt);
        if(index==-1) {
            // no overlap with active gt
            if(detections[k].check(anno.posROI)==-1) {
                // no overlap with any gt
                // false positive
                train.negROI.push_back(detections[k].bb);
                detections[k].verified = HYPO_FALSE_POS;
                ++count_errors;
            } else {
                // false positive, but only due to double detection
                detections[k].verified = HYPO_FALSE_POS_MULTI;
                ++count_errors;
            }
        }

        else {
            // true positive / remove gt
            gt.erase(gt.begin()+index);
            train.posROI.push_back(detections[k].bb);
            detections[k].verified = HYPO_CORRECT_POS;
        }
    }

    // false negatives (missed gt)
    for(unsigned int k=0;k<gt.size();++k) {
        train.posROI.push_back(gt[k]);
        Hypothesis hyp;
        hyp.conf = -1;
        hyp.error_score = 0;
        hyp.bb = gt[k];
        hyp.verified = HYPO_FALSE_NEG;
        detections.push_back(hyp);
        ++count_errors;
    }

    // blow threshold unseen estimates
    for(; k<end_k; ++k) {

        if(detections[k].check(anno.posROI)==-1) {
            // no overlap with any gt -> true negative
            if(train.negROI.size() < gt.size()+train.posROI.size())
            train.negROI.push_back(detections[k].bb);
            detections[k].verified = HYPO_CORRECT_NEG;
        }
        else {
            if(detections[k].check(gt)==-1) {
                // overlap with gt but not with active gt
                detections[k].verified = HYPO_CORRECT_NEG_MULTI;
            }
            else {
                // overlap with active gt
                detections[k].verified = HYPO_NEG_OVERLAP;
            }
        }
    }


    //cout << "Collect Training ROI: " << train.posROI.size() << " " << train.negROI.size() << endl;

    train.image_name = anno.image_name;
    train.sample_background = false;

    return count_errors;
}

int Hypothesis::check(vector<Rect>& posROI) {
    double measure = 0.5;
    int index = -1;

    for(unsigned int i=0;i<posROI.size();++i) {
        double tmp = iou(posROI[i]);
        if(tmp>measure) {
            measure = tmp;
            index = i;
        }
    }

    return index;
}

double Hypothesis::iou(Rect& gt) {
    Rect inter = bb & gt;
    double area_i = inter.width*inter.height;
    // intersection over union
    return area_i / (bb.width*bb.height + gt.width*gt.height - area_i);
}

void Hypotheses::create(Annotation& anno) {
    detections.resize(anno.posROI.size());
    used_for_train = true;
    for(unsigned int i=0;i<detections.size();++i) {
        detections[i].bb = anno.posROI[i];
        detections[i].conf = -1;
        detections[i].error_score = -1;
        detections[i].verified = HYPO_FALSE_NEG;
    }
}

// save detections to file
bool Hypotheses::save_detections(const char* detection_fname) {

    bool ok = false;

    ofstream out(detection_fname);
    if(out.is_open()) {

        out << detections.size() << " ";
        out << threshold << " " << error_score << " " << (int)used_for_train << endl;

        for(unsigned int i=0; i<detections.size(); ++i) {
            out << detections[i].conf << " ";
            out << detections[i].bb.x << " ";
            out << detections[i].bb.y << " ";
            out << detections[i].bb.width << " ";
            out << detections[i].bb.height << " ";
            out << detections[i].error_score << " ";
            out << (int)detections[i].verified << endl;
        }

        ok = true;

    }
    else {
        cerr << "Could not write detections " << detection_fname << endl;
    }

    return ok;

}


// show detections
void Hypotheses::show_detections(Mat& img, double threshold) {

    namedWindow( "Detections", CV_WINDOW_AUTOSIZE );
    Mat show;
    img.convertTo(show, CV_8UC3);
    for(int k=detections.size()-1; k>=0; --k) {
        if(detections[k].conf >= threshold)
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(10, 150*detections[k].conf, 10), 2);
    }
    imshow( "Detections", show );
    waitKey(0);

}


// show detections
void Hypotheses::show_detections(Mat& img) {

    namedWindow( "Detections", CV_WINDOW_AUTOSIZE );
    Mat show;
    img.convertTo(show, CV_8UC3);
    for(int k=detections.size()-1; k>=0; --k) {
        switch (detections[k].verified) {
            case HYPO_UNCHECKED: //white 1
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(255, 200.0*detections[k].conf/detections[0].conf, 255), 1);
            break;
            case  HYPO_CORRECT_POS: //green 2
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(100, 255, 100), 2);
            break;
            case HYPO_FALSE_POS: // red 2
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(255, 100, 100), 2);
            break;
            case  HYPO_FALSE_POS_MULTI: // red 1
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(255, 100, 100), 1);
            break;
            case HYPO_FALSE_NEG: // blue 2
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(100, 100, 255), 2);
            break;
            case HYPO_CORRECT_NEG: // black 2
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(50, 50, 50), 2);
            break;
            case  HYPO_CORRECT_NEG_MULTI: // black 1
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(50, 50, 50), 1);
            break;
            case  HYPO_NEG_OVERLAP: // gray 1
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(150, 150, 150), 1);
            break;
        }
    }
    imshow( "Detections", show );
    waitKey(0);

}

// show detections
void Hypotheses::show_detections2(Mat& img) {

    namedWindow( "Detections", CV_WINDOW_AUTOSIZE );
    Mat show;
    img.convertTo(show, CV_8UC3);
    for(int k=detections.size()-1; k>=0; --k) {
        switch (detections[k].verified) {
            case  HYPO_CORRECT_POS:
            // correct detection - green
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(100, 255, 100), 2);
            break;
            case  HYPO_FALSE_POS_MULTI:
            // wrong detection - red
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(255, 100, 100), 2);
            break;
            case HYPO_FALSE_POS:
            // wrong detection - red
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(255, 100, 100), 2);
            break;
            case HYPO_FALSE_NEG:
            // missed detection
            rectangle(show, Point(detections[k].bb.x,detections[k].bb.y), Point(detections[k].bb.x+detections[k].bb.width,detections[k].bb.y+detections[k].bb.height), CV_RGB(100, 100, 255), 2);
            break;
            default:
            // do not show
            break;
        }
    }
    imshow( "Detections", show );
    waitKey(0);

}
