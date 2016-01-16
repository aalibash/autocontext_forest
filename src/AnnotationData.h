/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef AnnotationDataH
#define AnnotationDataH

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

#include "Param.h"

struct Annotation {
    std::string image_name;
    std::vector<cv::Rect> posROI;
    std::vector<cv::Rect> negROI;
    bool sample_background;
    bool hard_neg;
};

class AnnotationData {
    public:
    bool loadAnnoFile(const char* filename);
    // get unit height/width and scales
    void getStatistics(StructParam* par, int i1 = 0, int i2 = -1);
    bool computeScale(StructParam* par, double min_scale, double max_scale);
    void reorder();

    std::vector<Annotation> AnnoData;
};

#endif
