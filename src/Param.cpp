/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "Param.h"

#include <libconfig.h++>
#include <iostream>

using namespace libconfig;
using namespace std;

// parses parameters for detection
bool StructParam::loadConfigDetect(const char* filename) {

  Config configFile;

  //try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }


  // look up parameter settings / values
  try{

    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // leaf id path
    leafId_path = (const char*)configFile.lookup("leafId_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");
    treepath_L2 = (const char*)configFile.lookup("treepath_L2");
    // Number of trees
    ntrees_L1 = configFile.lookup("number_of_trees_L1");
    ntrees_L2 = configFile.lookup("number_of_trees_L2");

    // Scale unit (percentage)
    u_scale = configFile.lookup("scale_unit");
    // Aspect Ratio Unit
    u_asp_ratio = configFile.lookup("asp_ratio_unit");
    // Scale increment
    d_scale = configFile.lookup("scale_increment");

    // Test file
    test_file = (const char*)configFile.lookup("test_file");
    // Path for storing hypotheses
    hypotheses_path = (const char*)configFile.lookup("hypotheses_path");

    // Stride for detection
    stride = configFile.lookup("stride");
    // Max. votes per leaf
    max_votes_leaf = configFile.lookup("max_votes_leaf");
    // Max. samples per leaf
    //max_samples_node = configFile.lookup("max_samples_node");
    // Reduction if too many samples per node percentage
    //reduce_samples_node = configFile.lookup("reduce_samples_node");
    // Threshold for detection
    d_thres = configFile.lookup("detect_threshold");
    // Max number of detection
    d_max = configFile.lookup("max_detection");
    // Minimum voting probability
    min_pfg_vote = configFile.lookup("min_positive_prob_vote");
    // Non-maxima suppression (percentage of bounding box)
    remove_vote = configFile.lookup("remove_vote_percent");

    // Learning mode LEARN_OFFLINE,LEARN_INCREMENTAL
    learn_mode = learn_type( (int)configFile.lookup("learning_mode") );

    // Scales
    for (int i = 0;i<configFile.lookup("scales").getLength();i++) {
      scales.push_back( configFile.lookup("scales")[i] );
      inv_scales.push_back(1.0/scales[i]);
    }
    // asp_ratios
    for (int i = 0;i<configFile.lookup("asp_ratios").getLength();i++) {
        double temp_asp_ratio = configFile.lookup("asp_ratios")[i];
        asp_ratios.push_back(temp_asp_ratio/u_asp_ratio);
        inv_asp_ratios.push_back(u_asp_ratio/temp_asp_ratio);
    }

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  //cout << endl << "------------------------------------" << endl << endl;
  //cout << "Trees:            " << ntrees << " " << treepath << endl;
  //cout << "Patches:          " << p_width << " " << p_height << endl;
  //cout << "Unit size:        " << u_width << " " << u_height << endl;
  //cout << "Scales:           "; for(unsigned int i=0;i<scales.size();++i) cout << scales[i] << " "; cout << endl;
  //cout << "Stride:           " << stride << endl;
  //cout << "Detection:        " << d_thres << " " << d_max << endl;
  //cout << endl << "------------------------------------" << endl << endl;

  return true;

}

// parses parameters for detection
bool StructParam::loadConfigDetect_L2(const char* filename) {

  Config configFile;

  //try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }


  // look up parameter settings / values
  try{

    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // leaf id path
    leafId_path = (const char*)configFile.lookup("leafId_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");
    treepath_L2 = (const char*)configFile.lookup("treepath_L2");
    // Number of trees
    ntrees_L1 = configFile.lookup("number_of_trees_L1");
    ntrees_L2 = configFile.lookup("number_of_trees_L2");

    // Scale unit (percentage)
    u_scale = configFile.lookup("scale_unit");
    // Aspect Ratio Unit
    u_asp_ratio = configFile.lookup("asp_ratio_unit");
    // Scale increment
    d_scale = configFile.lookup("scale_increment");

    // Test file
    test_file = (const char*)configFile.lookup("test_file");
    // Path for storing hypotheses
    hypotheses_path = (const char*)configFile.lookup("hypotheses_path");

    // Stride for detection
    stride = configFile.lookup("stride");
    // Max. votes per leaf
    max_votes_leaf = configFile.lookup("max_votes_leaf");
    // Max. samples per leaf
    //max_samples_node = configFile.lookup("max_samples_node");
    // Reduction if too many samples per node percentage
    //reduce_samples_node = configFile.lookup("reduce_samples_node");
    // Threshold for detection
    d_thres = configFile.lookup("detect_threshold");
    // Max number of detection
    d_max = configFile.lookup("max_detection");
    // Minimum voting probability
    min_pfg_vote = configFile.lookup("min_positive_prob_vote");
    // Non-maxima suppression (percentage of bounding box)
    remove_vote = configFile.lookup("remove_vote_percent");

    // Learning mode LEARN_OFFLINE,LEARN_INCREMENTAL
    learn_mode = learn_type( (int)configFile.lookup("learning_mode") );

    // Scales
    for (int i = 0;i<configFile.lookup("scales").getLength();i++) {
      scales.push_back( configFile.lookup("scales")[i] );
      inv_scales.push_back(1.0/scales[i]);
    }
    // asp_ratios
    for (int i = 0;i<configFile.lookup("asp_ratios").getLength();i++) {
        double temp_asp_ratio = configFile.lookup("asp_ratios")[i];
        asp_ratios.push_back(temp_asp_ratio/u_asp_ratio);
        inv_asp_ratios.push_back(u_asp_ratio/temp_asp_ratio);
    }

    // group related
    for (int i = 0;i<configFile.lookup("grp_size_x").getLength();i++) {
        grp_size_x.push_back( configFile.lookup("grp_size_x")[i] );
    }
    for (int i = 0;i<configFile.lookup("grp_size_y").getLength();i++) {
        grp_size_y.push_back( configFile.lookup("grp_size_y")[i] );
    }
    // L1 depth for leafIdMaps
    L1_depth = configFile.lookup("L1_depth");
    // lambda values
    for (int i = 0;i<configFile.lookup("lambda").getLength();i++) {
      lambda.push_back( configFile.lookup("lambda")[i] );
    }

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  //cout << endl << "------------------------------------" << endl << endl;
  //cout << "Trees:            " << ntrees << " " << treepath << endl;
  //cout << "Patches:          " << p_width << " " << p_height << endl;
  //cout << "Unit size:        " << u_width << " " << u_height << endl;
  //cout << "Scales:           "; for(unsigned int i=0;i<scales.size();++i) cout << scales[i] << " "; cout << endl;
  //cout << "Stride:           " << stride << endl;
  //cout << "Detection:        " << d_thres << " " << d_max << endl;
  //cout << endl << "------------------------------------" << endl << endl;

  return true;

}

bool StructParam::loadConfigFeature(const char* filename) {

  Config configFile;

  // try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }

  // look up parameter settings / values
  try{

    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Test file
    test_file = (const char*)configFile.lookup("test_file");
    // Scales
    for (int i = 0;i<configFile.lookup("scales").getLength();i++) {
      scales.push_back( configFile.lookup("scales")[i] );
      inv_scales.push_back(1.0/scales[i]);
    }

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  return true;

}


// parses parameters for training
bool StructParam::loadConfigTrain(const char* filename) {

  Config configFile;

  // try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }

  // look up parameter settings / values
  try{

    // Annotation file
    anno_file = (const char*)configFile.lookup("train_file");
    // leaf id path
    leafId_path = (const char*)configFile.lookup("leafId_path");
    // Number of training examples
    training_examples = configFile.lookup("training_examples");
    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");
    treepath_L2 = (const char*)configFile.lookup("treepath_L2");
    // Number of trees
    ntrees_L1 = configFile.lookup("number_of_trees_L1");
    ntrees_L2 = configFile.lookup("number_of_trees_L2");
    // Patch width
    p_width = configFile.lookup("patch_width");
    // Patch height
    p_height = configFile.lookup("patch_height");
    // Unit width in pixels
    //u_width = (int)configFile.lookup("unit_width");
    // Unit height in pixels
    //u_height = (int)configFile.lookup("unit_height");

    // Scale unit (percentage)
    u_scale = configFile.lookup("scale_unit");
    // Aspect Ratio Unit
    u_asp_ratio = configFile.lookup("asp_ratio_unit");
    // Scale increment
    d_scale = configFile.lookup("scale_increment");
    // Stride for detection
    stride = configFile.lookup("stride");
    // Number of samples from positive BB
    samples_bb_pos = configFile.lookup("samples_boundingboxes_positive");
    // Number of samples from negative BB
    samples_bb_neg = configFile.lookup("samples_boundingboxes_negative");
    // Number of samples from background
    samples_bg_neg = configFile.lookup("samples_background_negative");
    // Percentage of images to read with pos. bounding boxes
    samples_images_pos = configFile.lookup("samples_images_positive");
    // Percentage of images to read with neg. bounding boxes
    samples_images_neg = configFile.lookup("samples_images_negative");

    // Maximal depth of trees
    max_depth = configFile.lookup("maximum_tree_depth");
    // Minimum number of samples per leave
    min_samples_leaf = configFile.lookup("minimum_samples_leaf");
    // Number of tests per node
    num_test_node = configFile.lookup("number_tests_node");
    // Number of thresholds per test
    num_thres_test = configFile.lookup("number_thresholds_test");
    // Margin for splitting - information gain
    inf_margin = configFile.lookup("min_margin_inf_gain_test");
    // Margin for splitting - distance gain
    dist_margin = configFile.lookup("min_margin_dist_gain_test");

    // Max. votes per leaf
    max_votes_leaf = configFile.lookup("max_votes_leaf");
    // Max. samples per leaf
    max_samples_node = configFile.lookup("max_samples_node");
    // Reduction if too many samples per node percentage
    reduce_samples_node = configFile.lookup("reduce_samples_node");

    // Learning mode LEARN_OFFLINE,LEARN_INCREMENTAL
    learn_mode = learn_type( (int)configFile.lookup("learning_mode") );

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  return true;

}

// parses parameters for training
bool StructParam::loadConfigTrain_L2(const char* filename) {

  Config configFile;

  // try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }

  // look up parameter settings / values
  try{

    // Annotation file
    anno_file = (const char*)configFile.lookup("train_file");
    // leaf id path
    leafId_path = (const char*)configFile.lookup("leafId_path");
    // Number of training examples
    training_examples = configFile.lookup("training_examples");
    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");
    treepath_L2 = (const char*)configFile.lookup("treepath_L2");
    // Number of trees
    ntrees_L1 = configFile.lookup("number_of_trees_L1");
    ntrees_L2 = configFile.lookup("number_of_trees_L2");
    // Patch width
    p_width = configFile.lookup("patch_width");
    // Patch height
    p_height = configFile.lookup("patch_height");
    // group related
    for (int i = 0;i<configFile.lookup("grp_size_x").getLength();i++) {
        grp_size_x.push_back( configFile.lookup("grp_size_x")[i] );
    }
    for (int i = 0;i<configFile.lookup("grp_size_y").getLength();i++) {
        grp_size_y.push_back( configFile.lookup("grp_size_y")[i] );
    }
    // L1 depth for leafIdMaps
    L1_depth = configFile.lookup("L1_depth");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");

    // Scale unit (percentage)
    u_scale = configFile.lookup("scale_unit");
    // Aspect Ratio Unit
    u_asp_ratio = configFile.lookup("asp_ratio_unit");
    // Scale increment
    d_scale = configFile.lookup("scale_increment");
    // Stride for detection
    stride = configFile.lookup("stride");
    // Number of samples from positive BB
    samples_bb_pos = configFile.lookup("samples_boundingboxes_positive");
    // Number of samples from negative BB
    samples_bb_neg = configFile.lookup("samples_boundingboxes_negative");
    // Number of samples from background
    samples_bg_neg = configFile.lookup("samples_background_negative");
    // Percentage of images to read with pos. bounding boxes
    samples_images_pos = configFile.lookup("samples_images_positive");
    // Percentage of images to read with neg. bounding boxes
    samples_images_neg = configFile.lookup("samples_images_negative");

    // Maximal depth of trees
    max_depth = configFile.lookup("maximum_tree_depth");
    // Minimum number of samples per leave
    min_samples_leaf = configFile.lookup("minimum_samples_leaf");
    // Number of tests per node
    num_test_node = configFile.lookup("number_tests_node");
    // Number of thresholds per test
    num_thres_test = configFile.lookup("number_thresholds_test");
    // Margin for splitting - information gain
    inf_margin = configFile.lookup("min_margin_inf_gain_test");
    // Margin for splitting - distance gain
    dist_margin = configFile.lookup("min_margin_dist_gain_test");

    // Max. votes per leaf
    max_votes_leaf = configFile.lookup("max_votes_leaf");
    // Max. samples per leaf
    max_samples_node = configFile.lookup("max_samples_node");
    // Reduction if too many samples per node percentage
    reduce_samples_node = configFile.lookup("reduce_samples_node");

    // Learning mode LEARN_OFFLINE,LEARN_INCREMENTAL
    learn_mode = learn_type( (int)configFile.lookup("learning_mode") );

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  return true;

}

bool StructParam::loadConfigTrainDetect(const char* filename) {

  Config configFile;

  // try to read configuration file
  try {
    configFile.readFile(filename);
  }
  catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
  }
  catch(ParseException &pex) {
    cerr << "Parse error at " << filename << ":" << pex.getLine()
	 << " - " << pex.getError() << endl;
    return false;
  }

  // look up parameter settings / values
  try{

    // Annotation file
    //anno_file = (const char*)configFile.lookup("annotation_file");
    // Number of training examples
    training_examples = configFile.lookup("training_examples");
    // Path to images
    image_path = (const char*)configFile.lookup("image_path");
    depth_image_path = (const char*)configFile.lookup("depth_image_path");
    // Path to features (if pre-computed)
    feature_path = (const char*)configFile.lookup("feature_path");
    // Path to trees
    treepath_L1 = (const char*)configFile.lookup("treepath_L1");
    treepath_L2 = (const char*)configFile.lookup("treepath_L2");
    // Number of trees
    ntrees_L1 = configFile.lookup("number_of_trees_L1");
    // Patch width
    p_width = configFile.lookup("patch_width");
    // Patch height
    p_height = configFile.lookup("patch_height");
    // Unit width in pixels
    //u_width = (int)configFile.lookup("unit_width");
    // Unit height in pixels
    //u_height = (int)configFile.lookup("unit_height");

    // Scale unit (percentage)
    u_scale = configFile.lookup("scale_unit");
    // Scale increment
    d_scale = configFile.lookup("scale_increment");

    // Number of samples from positive BB
    samples_bb_pos = configFile.lookup("samples_boundingboxes_positive");
    // Number of samples from negative BB
    samples_bb_neg = configFile.lookup("samples_boundingboxes_negative");
    // Number of samples from background
    samples_bg_neg = configFile.lookup("samples_background_negative");
    // Percentage of images to read with pos. bounding boxes
    samples_images_pos = configFile.lookup("samples_images_positive");
    // Percentage of images to read with neg. bounding boxes
    samples_images_neg = configFile.lookup("samples_images_negative");

    // Maximal depth of trees
    max_depth = configFile.lookup("maximum_tree_depth");
    // Minimum number of samples per leave
    min_samples_leaf = configFile.lookup("minimum_samples_leaf");
    // Number of tests per node
    num_test_node = configFile.lookup("number_tests_node");
    // Number of thresholds per test
    num_thres_test = configFile.lookup("number_thresholds_test");
    // Margin for splitting - information gain
    inf_margin = configFile.lookup("min_margin_inf_gain_test");
    // Margin for splitting - distance gain
    dist_margin = configFile.lookup("min_margin_dist_gain_test");

    // Test file
    test_file = (const char*)configFile.lookup("test_file");
    // Path for storing hypotheses
    hypotheses_path = (const char*)configFile.lookup("hypotheses_path");

    // Stride for detection
    stride = configFile.lookup("stride");
    // Max. votes per leaf
    max_votes_leaf = configFile.lookup("max_votes_leaf");
    // Max. samples per leaf
    max_samples_node = configFile.lookup("max_samples_node");
    // Reduction if too many samples per node percentage
    reduce_samples_node = configFile.lookup("reduce_samples_node");
    // Threshold for detection
    d_thres = configFile.lookup("detect_threshold");
    // Max number of detection
    d_max = configFile.lookup("max_detection");
    // Minimum voting probability
    min_pfg_vote = configFile.lookup("min_positive_prob_vote");
    // Non-maxima suppression (percentage of bounding box)
    remove_vote = configFile.lookup("remove_vote_percent");

    // Process images chunk wise
    test_chunksize = configFile.lookup("test_chunksize");

    // Error model [FP,FN]
    for (int i = 0;i<configFile.lookup("error_model").getLength();i++) {
      C_error.push_back( configFile.lookup("error_model")[i] );
    }

    // Learning mode LEARN_OFFLINE,LEARN_INCREMENTAL
    learn_mode = learn_type( (int)configFile.lookup("learning_mode") );

  }
  catch(const SettingNotFoundException &nfex) {
    cerr << "Not found in configuration file!" << endl;
    return false;
  }

  return true;

}
