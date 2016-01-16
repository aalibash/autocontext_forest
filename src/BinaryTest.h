/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef BTestH
#define BTestH

#include "gsl/gsl_rng.h"
#include "opencv2/core/core.hpp"

#include "TrainingData.h"

struct BinaryTest_L1 {

    // evaluate
    bool evaluate(uchar** ptFCh, ushort** ptsFCh, int numEntrChar, int stepImg) const;
    bool evaluate(const std::vector<cv::Mat>& Patch) const;
    int evaluateValue(const std::vector<cv::Mat>& Patch) const;
    // generate
    void generate(gsl_rng* rng, int max_x, int max_y, int max_ch);
    void change_tao(int t) {tao=t;};

    void save(std::ofstream& out) const;
    void read(std::ifstream& in);
    void print() const;

private:

    int x1, x2, y1, y2, ch, tao;

};

struct BinaryTest_L2 {

    void change_tao(float t, const int& opt){ if(opt==0) leafTao = t;     // simple shotton
                                            if(opt==1) distTao = t;     // bhatt dist
                                            if(opt==2) leafTao_t2 = t;} // extended shotton

    void  save(std::ofstream& out) const;
    void  read(std::ifstream& in);
    void  print() const;

    bool  evaluate(const stHist& iPbDist);
    bool  evaluate(const std::vector<stHist>& iPbDist);
    float evaluateValue(const std::vector<stHist>& iPbDist) const;
    void  generate(gsl_rng* rng, const std::vector<stHist>& ipDist, const std::vector<std::vector<int> > idPool, const int& opt);

    int returnLeafId(){return leafId;};
    int returnTreeId(){return treeId;};

private:

    float         distTao, leafTao;
    int           leafId;
    int           treeId;
    int           testMode;
    stHist        rPbDist;
    // testId==2
    std::vector<int>   LeafId_t2;
    std::vector<float> IdWeight_t2;
    float       leafTao_t2;
};


#endif
