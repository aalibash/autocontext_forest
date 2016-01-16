/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef FPFNmodelH
#define FPFNmodelH

#include <vector>
#include <fstream>
#include <iostream>

struct GammaDist {

    GammaDist() {k=1;theta=100.0;};
    void set(std::vector<float>& x);
    double getPDF(double x);
    double getCDF(double x);

    double k;
    double theta;

};

class FPFNmodel {
    public:

    FPFNmodel() {p_pos = 0; C_error.resize(2,1);};
    void setErrorModel(std::vector<double>& C) {C_error = C;};
    void set(std::vector<float>& pos, std::vector<float>& neg);
    float getThreshold(float init_t);
    double score(float s, float thres);
    void save(std::ofstream& out);
    void print();

    private:

    double f(double t) {
    return C_error[0]*(1.0-p_pos)*(1-NegDist.getCDF(t)) + C_error[1]*p_pos*PosDist.getCDF(t);
    };
    double gradf(double t) {
    return ( C_error[0]*(p_pos-1.0)*NegDist.getPDF(t) + C_error[1]*p_pos*PosDist.getPDF(t) );
    };

    // Auxiliary functions
    double line_search(double t, double d);
    double GoldenSectionSearch(double a, double b, double c, double va, double vb, double vc, double t, double d);

    double p_pos;
    GammaDist NegDist;
    GammaDist PosDist;

    // [FP,FN]
    std::vector<double> C_error;

};

#endif

