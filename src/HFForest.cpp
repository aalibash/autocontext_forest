/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "HFForest.h"

#include <sys/time.h>
#include <stdio.h>
#include <fstream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// detection
void HFForest_L1::detect(string& fileName, Mat& originImg, Mat& depthImg, Hypotheses& hyp, string feature_name) {

    hyp.detections.clear();

    // timer
    timeval start, end;
    double runtime;

    // prepare voting space
    bigVote.resize(param->asp_ratios.size());
    for(unsigned int i=0; i<param->asp_ratios.size(); ++i){
        bigVote[i].resize(param->scales.size());
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // extract features
    vector<Features> Feat(param->scales.size());

    for(unsigned int ar=0; ar<param->asp_ratios.size(); ++ar){

        gettimeofday(&start, NULL);

        // allocate memory
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            bigVote[ar][k].create( int(originImg.rows * param->scales[k] + 0.5), int(originImg.cols*param->asp_ratios[ar]*param->scales[k] + 0.5), CV_32F);
        }

        #if 0
        //Debug
        namedWindow( "Show", CV_WINDOW_AUTOSIZE );
        imshow( "Show", originImg);
        waitKey(0);
        #endif

        for(unsigned int k=0;k<param->scales.size(); ++k) {

            if(feature_name.empty()) {

                if(depthImg.empty()){
                    Mat scaledImg;
                    resize(originImg, scaledImg, bigVote[ar][k].size());
                    Feat[k].extractFeatureChannels(scaledImg);
                }
                else{
                    Mat scaledImg, scaledDepthImg;
                    resize(originImg, scaledImg, bigVote[ar][k].size());
                    resize(depthImg, scaledDepthImg, bigVote[ar][k].size());
                    Feat[k].extractFeatureChannels16(scaledImg,scaledDepthImg);
                }

                #if 0
                //Debug
                namedWindow( "Show", CV_WINDOW_AUTOSIZE );
                imshow( "Show", scaledImg );
                waitKey(0);
                #endif

            } else {
                cout<<"software patch required for incorporating precomputed features"<<endl;
                exit(-1);
            }
        }

        gettimeofday(&end, NULL);
        runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
        cout << "Feature Extraction: " << runtime << " msec" << endl;
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        gettimeofday(&start, NULL);

        // vote
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            vote(bigVote[ar][k],Feat[k]);
        }

        gettimeofday(&end, NULL);
        runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
        cout << "Voting: " << runtime << " msec" << endl;
    }

    #if 0
    //Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    for(unsigned int ar=0;ar<param->asp_ratios.size(); ++ar) {
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            Mat show;
            bigVote[ar][k].convertTo(show, CV_8U, 200);
            imshow( "Show", show );
            Mat scaledImg;
            resize(originImg,scaledImg,bigVote[ar][k].size());
            imshow("originImg",scaledImg);
            waitKey(0);
        }
    }
    #endif

    gettimeofday(&start, NULL);

    // detect objects
    detect(hyp, bigVote);

    gettimeofday(&end, NULL);
    runtime = ( (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/(1000.0) );
    cout << "Detection: " << runtime << " msec" << endl;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}

// find detections in voting space
void HFForest::detect(Hypotheses& hyp, vector<vector<Mat> >& bigVote) {

    int aspNeighborhood = param->asp_ratios.size();
    int sclNeighborhood = param->scales.size();

    // scale to smallest scale
    vector< vector<Mat> > Hspace(param->asp_ratios.size(), vector<Mat>(param->scales.size()));
    for(unsigned int ar=0;ar<param->asp_ratios.size();++ar) {
        for(unsigned int k=0;k<param->scales.size();++k) {
            resize(bigVote[ar][k], Hspace[ar][k], bigVote[0][0].size());
            GaussianBlur(Hspace[ar][k], Hspace[ar][k], Size(5,5), 3);
        }
    }


    #if 0
    // Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    for(int ar=0;ar<param->asp_ratios.size(); ++ar) {
        for(int k=0;k<param->scales.size(); ++k) {
            Mat show;
            //Hspace[ar][k].convertTo(show, CV_8U, 200);
            bigVote[ar][k].convertTo(show, CV_8U, 20);
            imshow( "Show", show );
            waitKey(0);
        }
    }
    #endif


    // greedy detection
    int offh = int(param->u_height/2.0);
    int offw = int(param->u_width/2.0);
    int dh = int(0.5 * param->remove_vote * param->u_height * param->scales[0]);
    int dw = int(0.5 * param->remove_vote * param->u_width  * param->scales[0] * param->asp_ratios[0]);
    double conf;

    for(int i=0;i<param->d_max;++i) {

    vector<vector <double> > max_val(param->asp_ratios.size(), vector<double>(param->scales.size()));
    vector<vector <Point> > max_points(param->asp_ratios.size(),vector<Point>(param->scales.size()));
    int max_k = 0;
    int max_ar = 0;
    conf = -1;

    // find max
    for(unsigned int ar=0; ar<param->asp_ratios.size(); ++ar){
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            minMaxLoc(Hspace[ar][k], NULL, &max_val[ar][k], NULL, &max_points[ar][k]);
            if(max_val[ar][k] > conf) {
                conf = max_val[ar][k];
                max_k = k;
                max_ar = ar;
            }
        }
    }

    // conf larger than epsilon
    if(conf < 0.0001)
    break;

    double e_sc = param->inv_scales[max_k];
    double e_ar = param->inv_asp_ratios[max_ar];
    double e_x = max_points[max_ar][max_k].x;
    double e_y = max_points[max_ar][max_k].y;

    Hypothesis h;
    h.bb.x = int(e_x*param->inv_scales[0]*param->inv_asp_ratios[0]-offw*e_sc*e_ar+0.5); // we modified only the width NOT height
    h.bb.y = int(e_y*param->inv_scales[0]-offh*e_sc+0.5);
    h.bb.width  = int(2.0*offw*e_sc*e_ar+0.5); // we modified only width NOT height
    h.bb.height = int(2.0*offh*e_sc+0.5);
    h.conf = conf*float((param->stride)*(param->stride));
    h.verified = HYPO_UNCHECKED;

    hyp.detections.push_back(h);

    // mask detection
    int ax = int( e_x - dw*e_sc*e_ar + 0.5);
    int ay = int( e_y - dh*e_sc + 0.5);
    int bx = int( e_x + dw*e_sc*e_ar + 0.5);
    int by = int( e_y + dh*e_sc + 0.5);

    if(ax < 0) ax = 0;
    if(ay < 0) ay = 0;
    if(bx > Hspace[0][0].cols) bx = Hspace[0][0].cols;
    if(by > Hspace[0][0].rows) by = Hspace[0][0].rows;

    unsigned int startScIdx = max(0,max_k-sclNeighborhood);
    unsigned int endScIdx   = min(max_k+sclNeighborhood+1,(int)param->scales.size());
    unsigned int startArTdx = max(0,max_ar-aspNeighborhood);
    unsigned int endArIdx   = min(max_ar+aspNeighborhood+1,(int)param->asp_ratios.size());

    for(unsigned int ar=startArTdx; ar<endArIdx; ++ar){
        for(unsigned int k=startScIdx;k<endScIdx; ++k) {
            Hspace[ar][k]( Range(ay, by),Range(ax, bx) ) = Scalar(0);
        }
    }

    # if 0
    // Debug
    for(int ar=0;ar<Hspace.size(); ++ar) {
        for(int k=0;k<Hspace[ar].size(); ++k) {
            Mat show;
            Hspace[ar][k].convertTo(show, CV_8U, 10);
            imshow( "Show", show );
            waitKey(0);
        }
    }
    #endif

    }
}

// voting
void HFForest_L1::vote(Mat& Vote, Features& Feat) {

    Vote.setTo(0);

    int xoffset = param->p_width/2;
    int yoffset = param->p_height/2;

    int x, y, cx, cy; // x,y top left; cx,cy center of patch

    unsigned int u8Channels=0, u16Channels=0, cCount=0;
    for(unsigned int c=0; c<Feat.Channels.size(); ++c){
        if(Feat.Channels[c].elemSize()==1) u8Channels++;
        if(Feat.Channels[c].elemSize()==2) u16Channels++;
    }

    if(u8Channels+u16Channels != Feat.Channels.size()){
        cout<<"Data type not supported in vote()"<<endl;
        exit(-1);
    }

    uchar**  ptFCh;
    ushort** ptsFCh;

    if(u16Channels>0){
        ptFCh  = new uchar*[u8Channels];
        ptsFCh = new ushort*[u16Channels];
    }
    else if(u16Channels==0){
        ptFCh  = new uchar*[Feat.Channels.size()];
        ptsFCh = 0;
    }

    cy = yoffset;

    for(y=0; y<Feat.Channels[0].rows - param->p_height; y+=param->stride, cy+=param->stride) {

        // Get start of row
        if(u16Channels>0) {
            for(cCount=0; cCount<u8Channels; ++cCount){
                ptFCh[cCount] = Feat.Channels[cCount].ptr<uchar>(y);
            }
            for(cCount=u8Channels; cCount<Feat.Channels.size(); ++cCount){
                ptsFCh[cCount-u8Channels] = Feat.Channels[cCount].ptr<ushort>(y);
            }
        }
        else {
            for(unsigned int c=0; c<Feat.Channels.size(); ++c){
                ptFCh[c] = Feat.Channels[c].ptr<uchar>(y);
            }
        }

        cx = xoffset;

        for(x=0; x<Feat.Channels[0].cols - param->p_width; x+=param->stride, cx+=param->stride) {

            // regression for a single patch
            vector<LeafNode_L1*> result;
            regression(result, ptFCh, ptsFCh, u8Channels, Feat.Channels[0].step);

            //cout<<endl;
            // vote for all trees (leafs)
            for(vector<LeafNode_L1*>::iterator itL = result.begin(); itL!=result.end(); ++itL) {

                // To speed up the voting, one can vote only for patches
                // with a probability for foreground > 0.5
                //
                if((*itL)->pfg > param->min_pfg_vote) {

                    // limit votes
                    int vsize = min(param->max_votes_leaf,int((*itL)->vCenter.size()));

                    // voting weight for leaf
                    float w = (*itL)->pfg / float( vsize * result.size() );

                    // vote for all points stored in the leaf
                    for(int v=0; v < vsize; ++v) {

                        int wx = int(cx - (*itL)->vCenter[v].x + 0.5);
                        int wy = int(cy - (*itL)->vCenter[v].y + 0.5);
                        //cout << wx << "/" << wy << endl;
                        if(wy>=0 && wy<Vote.rows && wx>=0 && wx<Vote.cols) {
                            Vote.at<float>( wy, wx ) += w;
                        }
                    }
                } // end if
            }

            // increase pointer - x
            for(cCount=0; cCount<u8Channels; ++cCount){
                ptFCh[cCount]+=param->stride;
            }
            for(cCount=u8Channels; cCount<Feat.Channels.size(); ++cCount){
                ptsFCh[cCount-u8Channels]+=param->stride;
            }

        } // end for x
    } // end for y

    delete[] ptFCh;
    delete[] ptsFCh;

}

// detection
void HFForest_L1::returnVoteMaps(Mat& originImg, Mat& depthImg, vector<vector<Mat> >& bigVote) {

    // prepare voting space
    bigVote.resize(param->asp_ratios.size(),vector<Mat>(param->scales.size()));

    // extract features
    vector<Features> Feat(param->scales.size());

    for(unsigned int ar=0; ar<param->asp_ratios.size(); ++ar){

        // allocate memory
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            bigVote[ar][k].create(int(originImg.rows * param->scales[k] + 0.5), int(originImg.cols*param->asp_ratios[ar]*param->scales[k] + 0.5), CV_32F);
        }

        for(unsigned int k=0;k<param->scales.size(); ++k) {

            if(depthImg.empty()){
                Mat scaledImg;
                resize(originImg, scaledImg, bigVote[ar][k].size());
                Feat[k].extractFeatureChannels(scaledImg);
            }
            else{
                Mat scaledImg,scaledDepthImg;
                resize(originImg, scaledImg, bigVote[ar][k].size());
                resize(depthImg, scaledDepthImg, bigVote[ar][k].size());
                Feat[k].extractFeatureChannels16(scaledImg,scaledDepthImg);
            }



            #if 0
            //Debug
            namedWindow( "Show", CV_WINDOW_AUTOSIZE );
            imshow( "Show", scaledImg );
            waitKey(0);
            #endif

        }

        // vote
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            vote(bigVote[ar][k],Feat[k]);
        }

    }

    #if 0
    //Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    for(unsigned int ar=0;ar<param->asp_ratios.size(); ++ar) {
        for(unsigned int k=0;k<param->scales.size(); ++k) {
            Mat show;
            bigVote[ar][k].convertTo(show, CV_8U, 200);
            imshow( "Show", show );
            waitKey(0);
        }
    }
    #endif

}

void HFForest_L1::evaluateLeafIdMaps(cv::Mat& originImg, cv::Mat& depthImg, vector<vector<vector<cv::Mat> > >& leafIdMaps){

    leafIdMaps.resize(param->asp_ratios.size());
    for(unsigned int aspIdx=0; aspIdx<param->asp_ratios.size(); ++aspIdx){
        leafIdMaps[aspIdx].resize(param->scales.size());
        for(unsigned int sclIdx=0; sclIdx<param->scales.size(); ++sclIdx){

            // get the features after resizing
            Mat               scaledImg, scaledDepthImg;
            Features          feat;
            double scale    = param->scales[sclIdx];
            double aspRatio = param->asp_ratios[aspIdx];

            if(depthImg.empty()){
                resize(originImg,scaledImg,Size(int(originImg.cols*aspRatio*scale + 0.5), int(originImg.rows*scale + 0.5)));
                feat.extractFeatureChannels(scaledImg);
            }
            else{
                resize(originImg,scaledImg,Size(int(originImg.cols*aspRatio*scale + 0.5), int(originImg.rows*scale + 0.5)));
                resize(depthImg,scaledDepthImg,Size(int(originImg.cols*aspRatio*scale + 0.5), int(originImg.rows*scale + 0.5)));
                feat.extractFeatureChannels16(scaledImg,scaledDepthImg);
            }

            // get the id maps from all L1 trees
            getIdMaps(feat,leafIdMaps[aspIdx][sclIdx]);

//            Mat show;
//            leafIdMaps[aspIdx][sclIdx][0].convertTo(show,CV_8U,255*0.0005);
//            namedWindow("leafIdMap",CV_WINDOW_AUTOSIZE);
//            imshow("leafIdMap",show);
//            namedWindow("scaledImg",CV_WINDOW_AUTOSIZE);
//            imshow("scaledImg",scaledImg);
//            waitKey(0);

        }
    }
}

void HFForest_L1::evaluateTrainLeafIdMaps(cv::Mat& originImg, cv::Mat& depthImg, Annotation& anno, vector<vector<Mat> >& leafMappedTrainInstances){

    leafMappedTrainInstances.clear();

    if(anno.posROI.size()>0 && anno.hard_neg==0) {
        for(unsigned int idx=0; idx<anno.posROI.size(); ++idx){

            double scale = (double)param->u_height/ anno.posROI[idx].height;
            double aspRatio = (double)anno.posROI[idx].height / anno.posROI[idx].width;

            // resize the image to make the object of unit dimension
            Features feat;
            vector<Mat> tempLeafIdMaps;

            if(depthImg.empty()){
                Mat scaledImg;
                resize(originImg, scaledImg, Size(int(originImg.cols*scale*aspRatio/param->u_asp_ratio+0.5),int(originImg.rows*scale+0.5)));
                feat.extractFeatureChannels(scaledImg);
            }
            else{
                Mat scaledImg, scaledDepthImg;
                resize(originImg, scaledImg, Size(int(originImg.cols*scale*aspRatio/param->u_asp_ratio+0.5),int(originImg.rows*scale+0.5)));
                resize(depthImg, scaledDepthImg, Size(int(originImg.cols*scale*aspRatio/param->u_asp_ratio+0.5),int(originImg.rows*scale+0.5)));
                feat.extractFeatureChannels16(scaledImg,scaledDepthImg);
            }

            getIdMaps(feat,tempLeafIdMaps);
            leafMappedTrainInstances.push_back(tempLeafIdMaps);
        }
    }
    else {
        leafMappedTrainInstances.resize(1);
        Features feat;
        if(depthImg.empty()) {
            feat.extractFeatureChannels(originImg);
        }
        else {
            feat.extractFeatureChannels16(originImg,depthImg);
        }
        getIdMaps(feat,leafMappedTrainInstances[0]);
    }
}

// storing leaf node pointers for each patch
void HFForest_L1::getIdMaps(Features& Feat, vector<Mat>& idMap){

    int xoffset = param->p_width/2;
    int yoffset = param->p_height/2;
    int x, y, cx, cy; // x,y top left; cx,cy center of patch
    int iter=0;
    cy = yoffset;

    // array of pointers for features channels
    unsigned int u8Channels=0, u16Channels=0, cCount=0;
    for(unsigned int c=0; c<Feat.Channels.size(); ++c){
        if(Feat.Channels[c].elemSize()==1) u8Channels++;
        if(Feat.Channels[c].elemSize()==2) u16Channels++;
    }

    if(u8Channels+u16Channels != Feat.Channels.size()){
        cout<<"Data type not supported in vote()"<<endl;
        exit(-1);
    }

    // array of pointers for features channels
    uchar**  ptFCh;
    ushort** ptsFCh;

    if(u16Channels>0){
        ptFCh  = new uchar*[u8Channels];
        ptsFCh = new ushort*[u16Channels];
    }
    else if(u16Channels==0){
        ptFCh  = new uchar*[Feat.Channels.size()];
        ptsFCh = 0;
    }

    // init the idMap
    Mat_<int> tempMat(Feat.Channels[0].rows,Feat.Channels[0].cols);
    tempMat.setTo(-1);
    idMap.resize(vTrees.size());
    for(unsigned int i=0; i<vTrees.size(); ++i){
        tempMat.copyTo(idMap[i]);
    }

    for(y=0; y<Feat.Channels[0].rows - param->p_height; y+=param->stride, cy+=param->stride) {

        // Get start of row
        if(u16Channels>0){
            for(cCount=0; cCount<u8Channels; ++cCount){
                ptFCh[cCount] = Feat.Channels[cCount].ptr<uchar>(y);
            }
            for(cCount=u8Channels; cCount<Feat.Channels.size(); ++cCount){
                ptsFCh[cCount-u8Channels] = Feat.Channels[cCount].ptr<ushort>(y);
            }
        }
        else{
            for(unsigned int c=0; c<Feat.Channels.size(); ++c){
                ptFCh[c] = Feat.Channels[c].ptr<uchar>(y);
            }
        }
        cx = xoffset;

        for(x=0; x<Feat.Channels[0].cols - param->p_width; x+=param->stride, cx+=param->stride) {
            // regression for a single patch
            vector<LeafNode_L1*> result;
            regression(result, ptFCh, ptsFCh, u8Channels, Feat.Channels[0].step);

            // store the leaf information
            iter = 0;
            for(vector<LeafNode_L1*>::iterator itL = result.begin(); itL!=result.end(); ++itL) {

                if((*itL)->depth > param->L1_depth){

                    Node*     tmpN;
                    tmpN = (*itL)->parent;

                    while((*tmpN).depth > param->L1_depth){
                        tmpN = (*tmpN).parent;
                    }

                    (idMap[iter]).at<int>(cy,cx) = (*tmpN).node_id;
                }
                else{
                    (idMap[iter]).at<int>(cy,cx) = (*itL)->node_id;
                }

                ++iter;
            }

            // increase pointer - x
            for(cCount=0; cCount<u8Channels; ++cCount){
                ptFCh[cCount]+=param->stride;
            }
                for(cCount=u8Channels; cCount<Feat.Channels.size(); ++cCount){
                ptsFCh[cCount-u8Channels]+=param->stride;
            }
        }
    }

    delete[] ptFCh;
    delete[] ptsFCh;
}

void HFForest_L1::regression(vector<LeafNode_L1*>& result, uchar** ptFCh, ushort** ptsFCh, int numEntChar, int stepImg) {
    result.resize( vTrees.size() );
    for(int i=0; i<(int)vTrees.size(); ++i) {
        result[i] = vTrees[i].regression(ptFCh, ptsFCh, numEntChar, stepImg);
    }
}

void HFForest_L1::loadForest(std::string filename) {
    char buffer[1024];
    for(unsigned int i=0; i<vTrees.size(); ++i) {
        sprintf(buffer,"%s%03d.txt",filename.c_str(),i);
        vTrees[i].readTree(buffer, param);
    }
}

void HFForest_L1::saveForest(const char* filename, int index_offset) {
    for(unsigned int i=0; i<vTrees.size(); ++i) {
        saveForest(filename,i,index_offset);
    }
}

void HFForest_L1::saveForest(const char* filename, int index, int index_offset) {
    char buffer[1024];
    sprintf(buffer,"%s%03d.txt",filename,index+index_offset);
    vTrees[index].saveTree(buffer);
}

/*------------------------------------------------------------------------------------------------
---------------------------   DEFINITIONS FOR LAYER 2 BEGINS   -----------------------------------
------------------------------------------------------------------------------------------------*/

void HFForest_L2::loadForest(std::string filename) {

    char buffer[1024];
    for(unsigned int i=0; i<vTrees.size(); ++i) {
        sprintf(buffer,"%s%03dgroup.txt",filename.c_str(),i);
        vTrees[i].readTree(buffer, param);
    }
}

void HFForest_L2::saveForest(const char* filename, int index_offset) {
    for(unsigned int i=0; i<vTrees.size(); ++i) {
    saveForest(filename,i,index_offset);
    }
}

void HFForest_L2::saveForest(const char* filename, int index, int index_offset) {
    char buffer[1024];
    sprintf(buffer,"%s%03dgroup.txt",filename,index+index_offset);
    vTrees[index].saveTree(buffer);
}

void HFForest_L2::regression(std::vector<LeafNode_L2*>& result, stHist& ipDist) {
    result.resize( vTrees.size() );
    for(int i=0; i<(int)vTrees.size(); ++i) {
        result[i] = vTrees[i].regression(ipDist);
    }
}

void HFForest_L2::regression(std::vector<LeafNode_L2*>& result, vector<stHist>& ipDist) {
    result.resize( vTrees.size() );
    for(int i=0; i<(int)vTrees.size(); ++i) {
        result[i] = vTrees[i].regression(ipDist);
    }
}

void HFForest_L2::vote(vector<LeafNode_L2*> result, const int& y, const int& x, const int& gh, const int& gw, Mat& voteMap){
    LeafNode_L2* itL;

    for(unsigned int idx=0; idx<result.size(); ++idx){
        if(NULL!=result[idx]){
            itL = result[idx];
            if((*itL).pfg > param->min_pfg_vote) {
                // voting weight for leaf
                int vsize=0;
                for(unsigned int v=0; v < (*itL).vCenter.size(); ++v) {
                    if((*itL).vCenter[v].width==gw && (*itL).vCenter[v].height==gh){ ++vsize; }
                }
                vsize = min(param->max_votes_leaf,int((*itL).vCenter.size()));
                float w = (*itL).pfg/float(vsize);
                // vote for all points stored in the leaf
                vsize=0;
                for(unsigned int v=0; v < (*itL).vCenter.size(); ++v) {
                    if((*itL).vCenter[v].width==gw && (*itL).vCenter[v].height==gh){
                        int wx = int(x + (*itL).vCenter[v].x + 0.5);
                        int wy = int(y + (*itL).vCenter[v].y + 0.5);

                        if(wy>=0 && wy<voteMap.rows && wx>=0 && wx<voteMap.cols) {
                            voteMap.at<float>( wy, wx ) += w;
                        }
                        if(++vsize > param->max_votes_leaf) break;
                    }
                }
            }
        }
    }
}

void HFForest_L2::detect(vector<vector<vector<Mat> > >leafIdMap, Mat& originImg, Hypotheses& hyp) {

    // init Vote
    bigVote.resize(param->asp_ratios.size(),vector<Mat>(param->scales.size()));

    for(unsigned int aspIdx=0; aspIdx<param->asp_ratios.size(); ++aspIdx){
        for(unsigned int sclIdx=0; sclIdx<param->scales.size(); ++sclIdx){

            // perform the voting
            detect(leafIdMap[aspIdx][sclIdx], bigVote[aspIdx][sclIdx]);

#if 0
            Mat show;
            bigVote[aspIdx][sclIdx].convertTo(show,CV_8U,100);
            namedWindow("vote",CV_WINDOW_AUTOSIZE);
            imshow("vote",show);
            namedWindow("originImg",CV_WINDOW_AUTOSIZE);
            imshow("originImg",originImg);
            waitKey(0);
#endif

        }
    }
    detect(hyp,bigVote);
    cout<<"detections done"<<endl;
}


void HFForest_L2::detect(vector<Mat>& leafIdMap, Mat& voteMap){

    int gh, gw;
    stHist tempHist;
    vector<stHist> vTempHist;
    vector<LeafNode_L2*> result;
    int *lptr;

    voteMap.create(leafIdMap[0].rows,leafIdMap[0].cols,CV_32F);
    voteMap.setTo(0);

    // histogram for the whole image
    for(unsigned int temph=0; temph<param->grp_size_y.size(); ++temph){
        for(unsigned int tempw=0; tempw<param->grp_size_x.size(); ++tempw){

            gh = param->grp_size_y[temph];
            gw = param->grp_size_x[tempw];
//            cout<<gh<<" "<<gw<<endl;
            // visit each pixel location
            for(int y=param->p_height/2; y<leafIdMap[0].rows-param->p_height/2-gh; ++y){
                for(int x=param->p_width/2; x<leafIdMap[0].cols-param->p_width/2-gw; ++x){

// TODO: VERY UGLY. FIX THIS
#if 0
                    // FOR NODE-TEST MODES 1
                    for(unsigned int treeIdx=0; treeIdx<leafIdMap.size(); ++treeIdx){
                        // clear the previous histogram
                        tempHist.hist.clear();
                        tempHist.norm=0;
                        // visit the neighborhood
                        for(int ty=0; ty<gh; ++ty){
                            lptr = leafIdMap[treeIdx].ptr<int>(y+ty)+x;
                            for(int tx=0; tx<gw; ++tx){
                                tempHist.update(*lptr);
                                lptr++;
                            }
                        }
                        // sort and verify
                        tempHist.sort();
                        tempHist.verify();
                        result.clear();
                        regression(result,tempHist);
                        vote(result,y,x,gh,gw,voteMap);
                    }
#endif

#if 1
                    // FOR NODE-TEST MODE 0 or 2
                    vTempHist.clear();
                    vTempHist.resize(leafIdMap.size());
                    for(unsigned int treeIdx=0; treeIdx<leafIdMap.size(); ++treeIdx){
                        // clear the previous histogram
                        tempHist.hist.clear();
                        tempHist.norm=0;
                        // visit the neighborhood
                        for(int ty=0; ty<gh; ++ty){
                            lptr = leafIdMap[treeIdx].ptr<int>(y+ty)+x;
                            for(int tx=0; tx<gw; ++tx){

                                //if(*lptr<0) cout<< *lptr<<" ";
                                tempHist.update(*lptr);
                                lptr++;
                            }
                        }
                        // sort and verify
                        tempHist.sort();
                        tempHist.verify();
                        vTempHist[treeIdx] = tempHist;
                    }
                    result.clear();
                    regression(result,vTempHist);
                    vote(result,y,x,gh,gw,voteMap);
#endif

                }
            }
//            cout<<"one tree's hist image computed "<<endl;
//            Mat show;
//            voteMap.convertTo(show,CV_8U,10);
//            namedWindow("vote",CV_WINDOW_AUTOSIZE);
//            imshow("vote",show);
//            waitKey(0);
        }
    }
}

void HFForest_L2::returnVoteMaps(vector<vector<vector<Mat> > >leafIdMap, vector<vector<Mat> >& bigVote, Mat& originImg) {

    // init Vote
    bigVote.resize(param->asp_ratios.size(),vector<Mat>(param->scales.size()));

    for(unsigned int aspIdx=0; aspIdx<param->asp_ratios.size(); ++aspIdx){
        for(unsigned int sclIdx=0; sclIdx<param->scales.size(); ++sclIdx){

            // perform the voting
            detect(leafIdMap[aspIdx][sclIdx], bigVote[aspIdx][sclIdx]);

        }
    }
}

// running combinations on individual and group voting maps
void HFForest_L2::detect(vector<Hypotheses>& bigHyp, vector< vector <Mat> >& bigVote_L1, vector<vector<Mat> >& bigVote_L2) {

    bigHyp.clear();

    for(unsigned int lambdaIdx=0; lambdaIdx<param->lambda.size(); ++lambdaIdx){

        // merge the L1 and L2 votes based on lambda
        double lval           = param->lambda[lambdaIdx];
        double one_minus_lval = 1-lval;

        cout<<lval<<" "<<one_minus_lval<<endl;

        vector<vector<Mat> > tempBigVote;
        float *l1_ptr, *l2_ptr, *lcom_ptr;

        tempBigVote.resize(param->asp_ratios.size(), vector<Mat>(param->scales.size()));
        for(unsigned int ar=0; ar<param->asp_ratios.size(); ++ar){
            for(unsigned int k=0; k<param->scales.size(); ++k){

                // L1 and L2 size assert
//                cout<<bigVote_L1[ar][k].cols<<" "<<bigVote_L2[ar][k].cols<<" "<<bigVote_L1[ar][k].rows<<" "<<bigVote_L2[ar][k].rows<<endl;
                if(bigVote_L1[ar][k].size()!=bigVote_L2[ar][k].size()){
                    cout<<"error combining votemaps in HFForest::detect()"<<endl;
                    exit(-1);
                }

                tempBigVote[ar][k].create(bigVote_L1[ar][k].size(),CV_32F);
                for(int y=0; y<bigVote_L1[ar][k].rows; ++y){

                    l1_ptr   = bigVote_L1[ar][k].ptr<float>(y);
                    l2_ptr   = bigVote_L2[ar][k].ptr<float>(y);
                    lcom_ptr = tempBigVote[ar][k].ptr<float>(y);

                    for(int x=0; x<bigVote_L1[ar][k].cols; ++x, ++l1_ptr, ++l2_ptr, ++lcom_ptr){
                        *lcom_ptr = pow(*l1_ptr,one_minus_lval) * pow(*l2_ptr,lval);
//                        cout<<*lcom_ptr<<" "<<*l1_ptr<<endl;
                    }

                }
            }
        }

        Hypotheses hyp;
        detect(hyp,tempBigVote);
        bigHyp.push_back(hyp);
    }
}
