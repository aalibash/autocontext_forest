/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef FeaturesH
#define FeaturesH

#include "opencv2/core/core.hpp"

#include "HoG.h"

#define FEATURE_CHANNELS 15
#define extractFeatureChannels extractFeatureChannels15

class Features {

    public:
    void extractFeatureChannels32(cv::Mat& img);
    void extractFeatureChannels10(cv::Mat& img);
    void extractFeatureChannels15(cv::Mat& img);
    void extractFeatureChannels16(cv::Mat& img, cv::Mat& depthImg);
    bool loadFeatures(std::string name, double scale);
    void saveFeatures(std::string name, double scale);

    std::vector<cv::Mat> Channels;
    HoG hog;

    private:
    void maxfilt(cv::Mat &src, unsigned int width);
    void minfilt(cv::Mat &src, cv::Mat &dst, unsigned int width);

    void maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width);
    void maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
    void minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width);
    void minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width);
};

#endif
