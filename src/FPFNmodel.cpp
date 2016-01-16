/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "FPFNmodel.h"

#include <iostream>
#include <cmath>

#include "gsl/gsl_cdf.h"
#include "gsl/gsl_randist.h"

#define EPSILON_T 0.00001

using namespace std;

double GammaDist::getPDF(double x) {
    return gsl_ran_gamma_pdf(x, k, theta);
}

double GammaDist::getCDF(double x) {
    return gsl_cdf_gamma_P(x, k, theta);
}

void GammaDist::set(vector<float>& x) {

    if(x.size()==0) {
        k = 1;
        theta = 100;
    }

    else {

        // mean
        double mean = 0;
        for(unsigned int i=0;i<x.size();++i) {
            mean += x[i];
        }
        mean = mean/double(x.size());

        // variance
        double var = 0;
        for(unsigned int i=0;i<x.size();++i) {
            var += (x[i]-mean)*(x[i]-mean);
        }
        var = var/double(x.size());

        k = mean*mean/var;
        theta = var/mean;

    }

}

void FPFNmodel::set(vector<float>& pos, vector<float>& neg) {

    NegDist.set(neg);
    PosDist.set(pos);

    if(pos.size()==0 && neg.size()==0) {
        p_pos = 0;
    }
    else {
        p_pos = double(pos.size())/double(pos.size()+neg.size());
    }

};

float FPFNmodel::getThreshold(float init_t) {

    double t_old = init_t;
    double gt = gradf(t_old);
    double gt_old;
    double d = -gt;

    double lambda = 0;

    if(std::abs(gt)>0.0001) {
        lambda = line_search(t_old,d);
    }

    double t = t_old + lambda * d;

    while( std::abs(t_old-t) > EPSILON_T ) {

        t_old = t;
        gt_old = gt;

        gt = gradf(t);
        double beta = gt*gt/(gt_old*gt_old);

        d = -gt + beta*d;

        if( std::abs(gt) > EPSILON_T )
        lambda = line_search(t,d);
        else
        lambda = 0;

        t = t + lambda * d;

    }

    return (float)t;

}

double FPFNmodel::score(float s, float thres) {

    double p_n = (1-p_pos) * NegDist.getPDF(s);
    double p_p = p_pos     * PosDist.getPDF(s);
    double y;

    if(s>=thres) //FP
        y = C_error[0]*p_n/(p_n+p_p);
    else         //FN
        y = C_error[1]*p_p/(p_n+p_p);

    return y;
}

void FPFNmodel::print() {
    cout << "ErrorModel(Pos) " << PosDist.k << " " << PosDist.theta << endl;
    cout << "ErrorModel(Neg) " << NegDist.k << " " << NegDist.theta << endl;
}

void FPFNmodel::save(ofstream& out) {
    out << "ErrorModel(Pos) " << PosDist.k << " " << PosDist.theta << endl;
    out << "ErrorModel(Neg) " << NegDist.k << " " << NegDist.theta << endl;
}

double FPFNmodel::line_search(double t, double d) {

    double dl = 1;

    double la = 0;
    double va = f(t + la * d);

    double lb = dl;
    double vb = f(t + lb * d);

    double lc,vc;

    if(vb>va) {
        lc = lb;
        vc = vb;
        lb = dl/2.0;
        vb = f(t + lb * d);
    }
    else {
        lc = lb + dl;
        vc = f(t + lc * d);
        while(vb-vc > 0.00001) {
            la = lb;
            va = vb;
            lb = lc;
            vb = vc;
            lc = lc + dl;
            vc = f(t + lc * d);
        }
    }

    return GoldenSectionSearch(la, lb, lc, va, vb, vc, t,d);

}

double FPFNmodel::GoldenSectionSearch(double a, double b, double c, double va, double vb, double vc, double t, double d) {

    double phi = 1.618033989; //(1 + sqrt(5)) / 2;
    double resphi = 2 - phi;
    double x, vx;

    if (c - b > b - a) {
        x = b + resphi * (c - b);
    }
    else {
        x = b - resphi * (b - a);
    }


    if ( (std::abs(va-vb)<EPSILON_T && std::abs(vb-vc)<EPSILON_T) || std::abs(c - a) < EPSILON_T * (std::abs(b) + std::abs(c))) {
        x = (c + a) / 2;
    }
    else {
        vx = f(t + x * d);
        if(vx<vb) {
            if(c - b > b - a)
            x = GoldenSectionSearch(b, x, c, vb, vx, vc, t, d);
            else
            x = GoldenSectionSearch(a, x, b, va, vx, vb, t, d);
        } else {
            if(c - b > b - a)
            x = GoldenSectionSearch(a, b, x, va, vb, vx, t, d);
            else
            x = GoldenSectionSearch(x, b, c, vx, vb, vc, t, d);
        }
    }
    return x;
}


