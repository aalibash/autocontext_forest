/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef ParamH
#define ParamH

#include <string>
#include <vector>

#include "opencv2/core/core.hpp"

enum learn_type {LEARN_OFFLINE,LEARN_INCREMENTAL,LEARN_INCREMENTAL_PREDICTION};

// program parameters
struct StructParam {

    // Path to trees
    std::string treepath_L1;
    std::string treepath_L2;
    // leaf id path
    std::string leafId_path;
    // Number of trees
    int ntrees_L1;
    int ntrees_L2;
    // Patch width
    int p_width;
    // Patch height
    int p_height;
    // Unit height
    float u_width;
    // Unit height
    float u_height;
    // Scale Unit
    double u_scale;
    // Aspect Ratio Unit
    double u_asp_ratio;
    // Scale increment
    double d_scale;
    // Scales
    std::vector<double> scales;
    std::vector<double> inv_scales;
    // Aspect Ratios
    std::vector<double> asp_ratios;
    std::vector<double> inv_asp_ratios;
    // Stride
    int stride;
    // Max. votes per leaf
    int max_votes_leaf;
    // Max. samples per leaf
    int max_samples_node;
    // Reduction if too many samples per node percentage
    double reduce_samples_node;
    // Threshold detection
    float d_thres;
    // Maximal number of detections
    int d_max;
    // Minimum voting probability
    float min_pfg_vote;
    // Non-maxima suppression (percentage of bounding box)
    double remove_vote;

    // Anno file
    std::string anno_file;
    // Number of training examples
    int training_examples;
    // Path to images
    std::string image_path;
    std::string depth_image_path;
    // Path to features (if pre-computed)
    std::string feature_path;
    // Number of samples from positive BB
    int samples_bb_pos;
    // Number of samples from negative BB
    int samples_bb_neg;
    // Number of samples from background
    int samples_bg_neg;
    // Percentage of images to read with pos. bounding boxes
    double samples_images_pos;
    // Percentage of images to read with neg. bounding boxes
    double samples_images_neg;

    // Maximal depth of trees
    int max_depth;
    // Minimum number of samples per leave
    int min_samples_leaf;
    // Number of tests per node
    int num_test_node;
    // Number of thresholds per test
    int num_thres_test;
    // Margin for splitting - information gain
    double inf_margin;
    // Margin for splitting - distance gain
    double dist_margin;

    // Test file
    std::string test_file;
    // Path for storing hypotheses
    std::string hypotheses_path;

    // Process images chunk wise
    int test_chunksize;

    // Error model [FP,FN]
    std::vector<double> C_error;

    // Learning mode
    learn_type learn_mode;

    // grouping related
    // sizes list
    std::vector<int> grp_size_x;
    std::vector<int> grp_size_y;
    // depth of layer 1 to be considered
    int L1_depth;
    // range of lambda values
    std::vector<double> lambda;


    bool loadConfigDetect(const char* filename);
    bool loadConfigFeature(const char* filename);
    bool loadConfigTrain(const char* filename);
    bool loadConfigTrainDetect(const char* filename);

    bool loadConfigTrain_L2(const char* filename);
    bool loadConfigDetect_L2(const char* filename);

};

#endif
