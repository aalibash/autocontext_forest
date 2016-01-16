/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef HypothesesH
#define HypothesesH

#include <iostream>

#include "AnnotationData.h"

#include "gsl/gsl_randist.h"
#include "opencv2/core/core.hpp"

enum hypothesis_status {HYPO_UNCHECKED, HYPO_CORRECT_POS, HYPO_FALSE_POS, HYPO_FALSE_POS_MULTI, HYPO_FALSE_NEG, HYPO_CORRECT_NEG, HYPO_CORRECT_NEG_MULTI, HYPO_NEG_OVERLAP};

// detection hypothesis
struct Hypothesis {

    // bounding box
    cv::Rect bb;
    // confidence
    float conf;
    // BETA score
    double error_score;

    // evaluation
    hypothesis_status verified;

    int check(std::vector<cv::Rect>& posROI);
    double iou(cv::Rect& gt);

};

class Hypotheses {
    public:
        Hypotheses() {threshold = -1; error_score = -1;used_for_train=false;};
        void create(Annotation& anno);

        void show_detections(cv::Mat& img, double threshold);
        void show_detections(cv::Mat& img);
        void show_detections2(cv::Mat& img);
        bool save_detections(const char* detection_fname);

        // check with ground truth
        int check(Annotation& anno, double thres, Annotation& train);

        // threshold used for checking
        double threshold;

        // error score
        double error_score;

        // used for training
        bool used_for_train;

        // image name
        int annotation_index;

        std::vector<Hypothesis> detections;
};

#endif
