/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "BinaryTest.h"

#include <iostream>
#include <fstream>

using namespace std;

void BinaryTest_L1::generate(gsl_rng* rng, int max_x, int max_y, int max_ch) {
    x1 = gsl_rng_uniform_int(rng, max_x);
    y1 = gsl_rng_uniform_int(rng, max_y);
    x2 = gsl_rng_uniform_int(rng, max_x);
    y2 = gsl_rng_uniform_int(rng, max_y);
    ch = gsl_rng_uniform_int(rng, max_ch);
}

bool BinaryTest_L1::evaluate(uchar** ptFCh, ushort** ptsFCh, int numEntrChar, int stepImg) const {

    if(ch<numEntrChar){
        uchar* ptC = ptFCh[ch];
        // get pixel values
        int p1 = *(ptC+x1+y1*stepImg);
        int p2 = *(ptC+x2+y2*stepImg);
        // test
        return (( p1 - p2 ) >= tao);
    }
    else{
        ushort* ptC = ptsFCh[ch-numEntrChar];
        // get pixel values
        int p1 = *(ptC+x1+y1*stepImg);
        int p2 = *(ptC+x2+y2*stepImg);
        // test
        return (( p1 - p2 ) >= tao);
    }
}

bool BinaryTest_L1::evaluate(const std::vector<cv::Mat>& Patch) const {

    if(Patch[ch].elemSize()==1){
        cv::Mat C = Patch[ch];
        return ( (C.at<uchar>(y1,x1) - C.at<uchar>(y2,x2)) >= tao );
    }
    else if(Patch[ch].elemSize()==2){
        cv::Mat C = Patch[ch];
        return ( (int(C.at<unsigned short>(y1,x1)) - int(C.at<unsigned short>(y2,x2))) >= tao );
    }

    cout<<"error in BinaryTest_L1::evaluate()"<<endl;
    exit(-1);

}

int BinaryTest_L1::evaluateValue(const std::vector<cv::Mat>& Patch) const {

    if(Patch[ch].elemSize()==1){
        cv::Mat C = Patch[ch];
        return (C.at<uchar>(y1,x1) - C.at<uchar>(y2,x2));
    }
    else if(Patch[ch].elemSize()==2){
        cv::Mat C = Patch[ch];
        return (int(C.at<unsigned short>(y1,x1)) - int(C.at<unsigned short>(y2,x2)));
    }

    cout<<"error in BinaryTest_L1::evaluateValue()"<<endl;
    exit(-1);
}

void BinaryTest_L1::save(std::ofstream& out) const {
    out << x1 << " ";
    out << y1 << " ";
    out << x2 << " ";
    out << y2 << " ";
    out << ch << " ";
    out << tao << " ";
}

void BinaryTest_L1::print() const {
    cout << x1 << " ";
    cout << y1 << " ";
    cout << x2 << " ";
    cout << y2 << " ";
    cout << ch << " ";
    cout << tao << " " << flush;
}

void BinaryTest_L1::read(std::ifstream& in) {
    in >> x1;
    in >> y1;
    in >> x2;
    in >> y2;
    in >> ch;
    in >> tao;
    //print();
}

/*------------------------------------------------------------------------------------------------
---------------------------   DEFINITIONS FOR LAYER 2 BEGINS   -----------------------------------
------------------------------------------------------------------------------------------------*/

void BinaryTest_L2::generate(gsl_rng* rng, const vector<stHist>& ipDist, const vector<vector<int> > idPool, const int& opt){
    testMode = opt;

    if(opt==0){ // bin freq test
        treeId     = gsl_rng_uniform_int(rng,ipDist.size());

        //int binIdx = gsl_rng_uniform_int(rng,ipDist[treeId].hist.size());

        // select among the most frequently occuring bins
        int found=0;
        int binIdxIdx, binIdx, iterIdx=0;
        while(!found && iterIdx<20){
            iterIdx++;
            binIdxIdx = gsl_rng_uniform_int(rng,idPool[treeId].size());
            for(unsigned int idx=0; idx<ipDist[treeId].hist.size(); ++idx){
                if(ipDist[treeId].hist[idx].leafId==idPool[treeId][binIdxIdx]){
                    binIdx = idPool[treeId][binIdxIdx];
                    found=1;
                    break;
                }
            }
        }
        if(!found){
            binIdx = gsl_rng_uniform_int(rng,ipDist[treeId].hist.size());
        }
        leafId     = ipDist[treeId].hist[binIdx].leafId;
        leafTao    = ipDist[treeId].hist[binIdx].freq/ipDist[treeId].norm;
        //cout<<"leafId: "<<leafId<<" leafTao: "<<leafTao<<endl;
    }

    if(opt==1){ // bhattDist test
        treeId  = gsl_rng_uniform_int(rng,ipDist.size());
        rPbDist = ipDist[treeId];
        //cout<<"treeId: "<<treeId<<" hist size: "<<rPbDist.hist.size()<<" norm: "<<rPbDist.norm<<endl;
    }

    if(opt==2){
        int   binIdx;
        bool  found=0;

        leafTao_t2 = 0;
        LeafId_t2.resize(ipDist.size());
        IdWeight_t2.resize(ipDist.size());

        for(unsigned int treeIdx=0; treeIdx<ipDist.size(); ++treeIdx){
            // allocate a weight to the tree
            IdWeight_t2[treeIdx] = 1.0f/ipDist.size();
            // check if an entry from idPool exists
            for(unsigned int iterIdx=0; iterIdx<idPool[treeIdx].size(); ++iterIdx){
                binIdx = gsl_rng_uniform_int(rng,idPool[treeIdx].size());
                found  = 0;
                for(unsigned int idx=0; idx<ipDist[treeIdx].hist.size(); ++idx){
                    if(ipDist[treeIdx].hist[idx].leafId==idPool[treeIdx][binIdx]){
                        LeafId_t2[treeIdx] = ipDist[treeIdx].hist[idx].leafId;
                        leafTao_t2 += IdWeight_t2[treeIdx]*ipDist[treeIdx].hist[idx].freq/ipDist[treeIdx].norm;
                        found=1;
                        break;
                    }
                }
                if(found) break;
            }
            // else take a random entry from the histogram
            if(found==0){
                binIdx = gsl_rng_uniform_int(rng,ipDist[treeIdx].hist.size());
                LeafId_t2[treeIdx] = ipDist[treeIdx].hist[binIdx].leafId;
                leafTao_t2 += IdWeight_t2[treeIdx]*ipDist[treeIdx].hist[binIdx].freq/ipDist[treeIdx].norm;
            }
        }
    }
}

float BinaryTest_L2::evaluateValue(const vector<stHist>& tPbDist) const{

    if(testMode==0){ // bin freq test; simple shotton test

        for(unsigned int i=0; i<tPbDist[treeId].hist.size(); ++i){
            if(tPbDist[treeId].hist[i].leafId==leafId) {
                return (float)tPbDist[treeId].hist[i].freq/tPbDist[treeId].norm;
            }
        }

        return 0;
    }

    if(testMode==1){ // bhattDist test
        unsigned int tcnt=0;
        unsigned int rcnt=0;
        unsigned int tmax=tPbDist[treeId].hist.size();
        unsigned int rmax=rPbDist.hist.size();
        float        bhattDist=0;
        int          iter=0;
        int          maxIter=10000;

        while(tcnt<tmax && rcnt<rmax && iter<maxIter){

            if(tPbDist[treeId].hist[tcnt].leafId == rPbDist.hist[rcnt].leafId){
                bhattDist+=sqrt(tPbDist[treeId].hist[tcnt].freq*rPbDist.hist[rcnt].freq);
                tcnt++;
                rcnt++;
            }

            else if(tPbDist[treeId].hist[tcnt].leafId<rPbDist.hist[rcnt].leafId){tcnt++;}

            else{rcnt++;}
            iter++;
        }

        if(iter==maxIter || tPbDist[treeId].norm*rPbDist.norm==0){
            cout<<"suspicious behaviour in BinaryTest::evaluateValue()"<<endl;
            return 0;
        }

        bhattDist/=sqrt(tPbDist[treeId].norm*rPbDist.norm);

//        if(bhattDist==1){
//            cout<<endl<<endl;
//            stHist tempHist = tPbDist[treeId];
//            cout<<"test histogram: "<<endl;
//            tempHist.disp();
//            cout<<endl<<endl;
//            cout<<"ref histogram: "<<endl;
//            tempHist = rPbDist;
//            tempHist.disp();
//            cout<<endl<<endl;
//            cout<<"bhatt dist: "<<bhattDist<<endl;
//            cout<<endl<<endl;
//        }

        return bhattDist;
    }

    if(testMode==2){
        float tao=0;
        for(unsigned int treeIdx=0; treeIdx<tPbDist.size(); ++treeIdx){
            for(unsigned int idx=0; idx<tPbDist[treeIdx].hist.size(); ++idx){
                if(tPbDist[treeIdx].hist[idx].leafId==LeafId_t2[treeIdx]){
                    tao += IdWeight_t2[treeIdx]*tPbDist[treeIdx].hist[idx].freq/tPbDist[treeIdx].norm;
                    break;
                }
            }
        }
        return tao;
    }

    return 0;
}

bool BinaryTest_L2::evaluate(const vector<stHist>& iPbDist){

    if(testMode==2){
        float tao=0;
        for(unsigned int treeIdx=0; treeIdx<iPbDist.size(); ++treeIdx){
            for(unsigned int idx=0; idx<iPbDist[treeIdx].hist.size(); ++idx){
                if(iPbDist[treeIdx].hist[idx].leafId==LeafId_t2[treeIdx]){
                    tao += IdWeight_t2[treeIdx]*iPbDist[treeIdx].hist[idx].freq/iPbDist[treeIdx].norm;
                    break;
                }
            }
        }

        if(tao > leafTao_t2){return true;}
        return false;
    }

    if(testMode==0){ // bin freq test
//        cout<<iPbDist[treeId].hist.size()<<endl;
        for(unsigned int i=0; i<iPbDist[treeId].hist.size(); ++i){
            if(iPbDist[treeId].hist[i].leafId==leafId && ((float)iPbDist[treeId].hist[i].freq/iPbDist[treeId].norm)>=leafTao){
                return true;
            }
        }

        return false;
    }

    return false;
}

bool BinaryTest_L2::evaluate(const stHist& iPbDist){

    if(testMode==1){ // bhattDist test
        unsigned int tcnt=0;
        unsigned int rcnt=0;
        unsigned int tmax=iPbDist.hist.size();
        unsigned int rmax=rPbDist.hist.size();
        float        bhattDist=0;
        int          iter=0;
        int          maxIter=10000;

        while(tcnt<tmax && rcnt<rmax && iter<maxIter){

            if(iPbDist.hist[tcnt].leafId==rPbDist.hist[rcnt].leafId){
                bhattDist+=sqrt(iPbDist.hist[tcnt].freq*rPbDist.hist[rcnt].freq);
                tcnt++;
                rcnt++;
            }

            else if(iPbDist.hist[tcnt].leafId<rPbDist.hist[rcnt].leafId){tcnt++;}

            else{rcnt++;}
            iter++;
        }

        if(iter==maxIter || iPbDist.norm*rPbDist.norm==0){
            cout<<"suspicious behaviour in BinaryTest::evaluate()"<<endl;
        }

        bhattDist/=sqrt(1+iPbDist.norm*rPbDist.norm);
        if(bhattDist>distTao) {return true;}
        return false;
    }

    return false;
}

void BinaryTest_L2::save(std::ofstream& out) const {

    out<<(int)testMode<<" ";
    if(testMode==0){
        out<<treeId<<" ";
        out<<leafTao<<" ";
        out<<leafId<<" ";
    }
    if(testMode==1){
        out << treeId<<" ";
        out << distTao << " ";
        out << rPbDist.norm << " ";
        out << rPbDist.hist.size() << " ";
        for(unsigned int i=0; i<rPbDist.hist.size(); ++i){
            out<<rPbDist.hist[i].leafId<<" "<<rPbDist.hist[i].freq<<" ";
        }
    }
    if(testMode==2){
        out<<LeafId_t2.size()<<" ";
        for(unsigned int i=0; i<LeafId_t2.size(); ++i){
            out<<LeafId_t2[i]<<" "<<IdWeight_t2[i]<<" ";
    }
        out<<leafTao_t2<<" ";
    }
}

void BinaryTest_L2::print() const {

    if(testMode==0){
        cout << treeId <<" ";
        cout << leafId << " ";
        cout << leafTao << " ";
        cout<<flush;
    }
    if(testMode==1){
        cout << treeId <<" ";
        cout << distTao << " ";
        cout << rPbDist.norm << " ";
        cout << rPbDist.hist.size() << " ";
        for(unsigned int i=0; i<rPbDist.hist.size(); ++i){
            cout<<rPbDist.hist[i].leafId<<" "<<rPbDist.hist[i].freq<<" ";
        };
        cout<<flush;
    }
    if(testMode==2){
        cout<<LeafId_t2.size()<<" ";
        for(unsigned int i=0; i<LeafId_t2.size(); ++i){
            cout << LeafId_t2[i]<<" "<<IdWeight_t2[i]<<" ";
        }
        cout<<leafTao_t2<<" ";
        cout<<flush;
    }

}

void BinaryTest_L2::read(std::ifstream& in) {

    int temp;
    in >> temp;
    testMode = temp;

    if(testMode==0){
        in >> treeId;
        in >> leafTao;
        in >> leafId;
        rPbDist.hist.clear();
        rPbDist.norm = 0;
        distTao = 0;
    }
    if(testMode==1){
        int dummy;
        in >> treeId;
        in >> distTao;
        in >> rPbDist.norm;
        in >> dummy;
        rPbDist.hist.resize(dummy);
        for(int i=0; i<dummy; ++i){
            in >> rPbDist.hist[i].leafId;
            in >> rPbDist.hist[i].freq;
        }
        leafTao = 0;
        leafId = 0;
    }
    if(testMode==2){
        int   dummy;
        float dummyWt;
        in >> dummy;
        LeafId_t2.resize(dummy);
        IdWeight_t2.resize(dummy);
        for(unsigned int i=0; i<LeafId_t2.size(); ++i){
            in >> dummy;
            LeafId_t2[i]=dummy;
            in >> dummyWt;
            IdWeight_t2[i] = dummyWt;
        }
        in >> leafTao_t2;
    }
}
