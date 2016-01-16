/*
// Author: Abhilash Srikantha, MPI Tuebingen
// abhilash.srikantha@tue.mpg.de
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "Features.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <deque>
#include <stdio.h>

using namespace std;
using namespace cv;


bool Features::loadFeatures(string name, double scale) {
    bool ok = true;

    int s = scale*100;
    Channels.resize(FEATURE_CHANNELS);

    for(int i=0;i<FEATURE_CHANNELS;++i) {
        char buffer[1024];
        sprintf(buffer,"%s_C%02d_%02d_S%03d.pgm",name.c_str(),(int)Channels.size(),i,s);
        //cout << "Read feature file " << buffer << endl;
        Channels[i] = imread(buffer,0);
        if(Channels[i].empty()) {
            cout << "Could not read feature file " << buffer << endl;
            ok = false;
            Channels.clear();
            break;
        }
    }

    return ok;
}

void Features::saveFeatures(string name, double scale) {

    int s = scale*100;
    for(unsigned int i=0;i<Channels.size();++i) {
        char buffer[1024];
        sprintf(buffer,"%s_C%02d_%02d_S%03d.pgm",name.c_str(),(int)Channels.size(),i,s);
        imwrite(buffer,Channels[i]);
    }

}

void Features::extractFeatureChannels32(Mat& img) {
    // 32 feature channels
    // 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)
    // 16+16 channels: minfilter + maxfilter on 5x5 neighborhood

    // changed 32 to 16 (no filtering)

    Channels.resize(FEATURE_CHANNELS);
    for(int c=0; c<FEATURE_CHANNELS; ++c)
        Channels[c].create(img.rows,img.cols, CV_8U);
    Mat I_x;
    Mat I_y;

    // Get intensity
    cvtColor( img, Channels[0], CV_RGB2GRAY );

    // |I_x|, |I_y|
    Sobel(Channels[0],I_x,CV_16S,1,0,3);
    Sobel(Channels[0],I_y,CV_16S,0,1,3);

    convertScaleAbs( I_x, Channels[3], 0.25);
    convertScaleAbs( I_y, Channels[4], 0.25);

    int rows = I_x.rows;
    int cols = I_y.cols;

    if(I_x.isContinuous() && I_y.isContinuous() && Channels[1].isContinuous() && Channels[2].isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    for( int y = 0; y < rows; y++ ) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = Channels[1].ptr<uchar>(y);
        for( int x = 0; x < cols; x++ ) {
            // Avoid division by zero
            float tx = (float)ptr_Ix[x] + (float)copysign(0.000001f, (float)ptr_Ix[x]);
            // Scaling [-pi/2 pi/2] -> [0 80*pi]
            ptr_out[x]=saturate_cast<uchar>( ( atan((float)ptr_Iy[x]/tx)+3.14159265f/2.0f ) * 80 );
        }
    }

    // Magnitude of gradients
    for( int y = 0; y < rows; y++ ) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = Channels[2].ptr<uchar>(y);
        for( int x = 0; x < cols; x++ ) {
            ptr_out[x] = saturate_cast<uchar>( sqrt((float)ptr_Ix[x]*(float)ptr_Ix[x] + (float)ptr_Iy[x]*(float)ptr_Iy[x]) );
        }
    }

    // 9-bin HOG feature stored at vImg[7] - vImg[15]
    hog.extractOBin(Channels[1], Channels[2], Channels, 7);

    // |I_xx|, |I_yy|
    Sobel(Channels[0],I_x,CV_16S, 2,0,3);
    convertScaleAbs( I_x, Channels[5], 0.25);

    Sobel(Channels[0],I_y,CV_16S, 0,2,3);
    convertScaleAbs( I_y, Channels[6], 0.25);

    // L, a, b
    Mat imgRGB(img.size(), CV_8UC3);
    cvtColor( img, imgRGB, CV_RGB2Lab  );

    // Split color channels
    Mat out[] = { Channels[0], Channels[1], Channels[2] };
    int from_to[] = { 0,0 , 1,1, 2,2 };
    mixChannels( &imgRGB, 1, out, 3, from_to, 3 );

    // min filter
    for(int c=0; c<16; ++c)
    minfilt(Channels[c], Channels[c+16], 5);

    //max filter
    for(int c=0; c<16; ++c)
    maxfilt(Channels[c], 5);



#if 0
// Debug
namedWindow( "Show", CV_WINDOW_AUTOSIZE );
for(int i=0; i<FEATURE_CHANNELS; i++) {
imshow( "Show", Channels[i] );
waitKey(0);
}
#endif


}

void Features::extractFeatureChannels16(Mat& img, Mat& depthImg) {

  // 9 hog channels   - UINT8
  // 6 color channels - UINT8
  // 1 depth channel  - UINT16

  const int DEPTH_FEATURE_CHANNELS = 16;

  Channels.resize(DEPTH_FEATURE_CHANNELS);
  for(int c=0; c<DEPTH_FEATURE_CHANNELS-1; ++c){
    Channels[c].create(img.rows,img.cols, CV_8U);
  }
  Channels[DEPTH_FEATURE_CHANNELS-1].create(img.rows,img.cols,CV_16U);

  Mat I_x;
  Mat I_y;

  // Get intensity
  cvtColor( img, Channels[0], CV_RGB2GRAY );

  // |I_x|, |I_y|
  Sobel(Channels[0],I_x,CV_16S,1,0,3);
  Sobel(Channels[0],I_y,CV_16S,0,1,3);

  //convertScaleAbs( I_x, Channels[3], 0.25);
  //convertScaleAbs( I_y, Channels[4], 0.25);

  int rows = I_x.rows;
  int cols = I_y.cols;

  if(I_x.isContinuous() && I_y.isContinuous() && Channels[1].isContinuous() && Channels[2].isContinuous()) {
    cols *= rows;
    rows = 1;
  }

  for( int y = 0; y < rows; y++ ) {
    short* ptr_Ix = I_x.ptr<short>(y);
    short* ptr_Iy = I_y.ptr<short>(y);
    uchar* ptr_out = Channels[1].ptr<uchar>(y);
    for( int x = 0; x < cols; x++ )
    {
      // Avoid division by zero
      float tx = (float)ptr_Ix[x] + (float)copysign(0.000001f, (float)ptr_Ix[x]);
      // Scaling [-pi/2 pi/2] -> [0 80*pi]
      ptr_out[x]=saturate_cast<uchar>( ( atan((float)ptr_Iy[x]/tx)+3.14159265f/2.0f ) * 80 );
    }
  }

  // Magnitude of gradients
  for( int y = 0; y < rows; y++ ) {
    short* ptr_Ix = I_x.ptr<short>(y);
    short* ptr_Iy = I_y.ptr<short>(y);
    uchar* ptr_out = Channels[2].ptr<uchar>(y);
    for( int x = 0; x < cols; x++ )
    {
      ptr_out[x] = saturate_cast<uchar>( sqrt((float)ptr_Ix[x]*(float)ptr_Ix[x] + (float)ptr_Iy[x]*(float)ptr_Iy[x]) );
    }
  }

  // 9-bin HOG feature stored at vImg[7] - vImg[15]
  hog.extractOBin(Channels[1], Channels[2], Channels, 3);

  // L, a, b
  Mat imgRGB(img.size(), CV_8UC3);
  cvtColor( img, imgRGB, CV_RGB2Lab  );

  // Split color channels
  Mat out[] = { Channels[0], Channels[1], Channels[2] };
  int from_to[] = { 0,0 , 1,1, 2,2 };
  mixChannels( &imgRGB, 1, out, 3, from_to, 3 );

  // min filter
  for(int c=0; c<3; ++c)
    minfilt(Channels[c], Channels[c+12], 5);

  //max filter
  for(int c=0; c<12; ++c)
    maxfilt(Channels[c], 5);

  // Get Depth
  depthImg.copyTo(Channels[DEPTH_FEATURE_CHANNELS-1]);

#if 0
  // Debug
  namedWindow( "Show", CV_WINDOW_AUTOSIZE );
  for(int i=0; i<FEATURE_CHANNELS-1; i++) {
    imshow( "Show", Channels[i] );
    waitKey(0);
  }
  cout<<"cannot imshow the 16bit depth data"<<endl;
#endif


}

void Features::extractFeatureChannels15(Mat& img) {
    // 9 feature channels
    // 3+9 channels: Lab + HOGlike features with 9 bins

    Channels.resize(FEATURE_CHANNELS);
    for(int c=0; c<FEATURE_CHANNELS; ++c)
    Channels[c].create(img.rows,img.cols, CV_8U);
    Mat I_x;
    Mat I_y;

    // Get intensity
    cvtColor( img, Channels[0], CV_RGB2GRAY );

    // |I_x|, |I_y|
    Sobel(Channels[0],I_x,CV_16S,1,0,3);
    Sobel(Channels[0],I_y,CV_16S,0,1,3);

    //convertScaleAbs( I_x, Channels[3], 0.25);
    //convertScaleAbs( I_y, Channels[4], 0.25);

    int rows = I_x.rows;
    int cols = I_y.cols;

    if(I_x.isContinuous() && I_y.isContinuous() && Channels[1].isContinuous() && Channels[2].isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    for( int y = 0; y < rows; y++ ) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = Channels[1].ptr<uchar>(y);
        for( int x = 0; x < cols; x++ ) {
            // Avoid division by zero
            float tx = (float)ptr_Ix[x] + (float)copysign(0.000001f, (float)ptr_Ix[x]);
            // Scaling [-pi/2 pi/2] -> [0 80*pi]
            ptr_out[x]=saturate_cast<uchar>( ( atan((float)ptr_Iy[x]/tx)+3.14159265f/2.0f ) * 80 );
        }
    }

    // Magnitude of gradients
    for( int y = 0; y < rows; y++ ) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = Channels[2].ptr<uchar>(y);
        for( int x = 0; x < cols; x++ ) {
            ptr_out[x] = saturate_cast<uchar>( sqrt((float)ptr_Ix[x]*(float)ptr_Ix[x] + (float)ptr_Iy[x]*(float)ptr_Iy[x]) );
        }
    }

    // 9-bin HOG feature stored at vImg[7] - vImg[15]
    hog.extractOBin(Channels[1], Channels[2], Channels, 3);

    // L, a, b
    Mat imgRGB(img.size(), CV_8UC3);
    cvtColor( img, imgRGB, CV_RGB2Lab  );

    // Split color channels
    Mat out[] = { Channels[0], Channels[1], Channels[2] };
    int from_to[] = { 0,0 , 1,1, 2,2 };
    mixChannels( &imgRGB, 1, out, 3, from_to, 3 );

    // min filter
    for(int c=0; c<3; ++c)
    minfilt(Channels[c], Channels[c+12], 5);

    //max filter
    for(int c=0; c<12; ++c)
    maxfilt(Channels[c], 5);



    #if 0
    // Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    for(int i=0; i<FEATURE_CHANNELS; i++) {
    imshow( "Show", Channels[i] );
    waitKey(0);
    }
    #endif

}

void Features::extractFeatureChannels10(Mat& img) {
    // 10 feature channels
    // 7 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|
    // 3+7 channels: minfilter (L,a,b) + maxfilter (all) on 5x5 neighborhood

    Channels.resize(FEATURE_CHANNELS);
    for(int c=0; c<FEATURE_CHANNELS; ++c)
    Channels[c].create(img.rows,img.cols, CV_8U);
    Mat I_x;
    Mat I_y;

    // Get intensity
    cvtColor( img, Channels[0], CV_RGB2GRAY );

    // |I_x|, |I_y|
    Sobel(Channels[0],I_x,CV_16S,1,0,3);
    Sobel(Channels[0],I_y,CV_16S,0,1,3);

    convertScaleAbs( I_x, Channels[3], 0.25);
    convertScaleAbs( I_y, Channels[4], 0.25);

    // |I_xx|, |I_yy|
    Sobel(Channels[0],I_x,CV_16S, 2,0,3);
    convertScaleAbs( I_x, Channels[5], 0.25);

    Sobel(Channels[0],I_y,CV_16S, 0,2,3);
    convertScaleAbs( I_y, Channels[6], 0.25);

    // L, a, b
    Mat imgRGB(img.size(), CV_8UC3);
    cvtColor( img, imgRGB, CV_RGB2Lab  );

    // Split color channels
    Mat out[] = { Channels[0], Channels[1], Channels[2] };
    int from_to[] = { 0,0 , 1,1, 2,2 };
    mixChannels( &imgRGB, 1, out, 3, from_to, 3 );

    // min filter
    for(int c=0; c<3; ++c)
    minfilt(Channels[c], Channels[c+7], 5);

    //max filter
    for(int c=0; c<7; ++c)
    maxfilt(Channels[c], 5);

    #if 0
    // Debug
    namedWindow( "Show", CV_WINDOW_AUTOSIZE );
    for(int i=0; i<FEATURE_CHANNELS; i++) {
        imshow( "Show", Channels[i] );
        waitKey(0);
    }
    #endif

}

void Features::maxfilt(Mat &src, unsigned int width) {

    unsigned int step = src.step;
    uchar* s_data = src.ptr<uchar>(0);

    for(int  y = 0; y < src.rows; y++) {
        maxfilt(s_data+y*step, 1, src.cols, width);
    }

    s_data = src.ptr<uchar>(0);

    for(int  x = 0; x < src.cols; x++)
        maxfilt(s_data+x, step, src.rows, width);

}

void Features::minfilt(Mat &src, Mat &dst, unsigned int width) {

    unsigned int step = src.step;
    uchar* s_data = src.ptr<uchar>(0);
    uchar* d_data = dst.ptr<uchar>(0);

    for(int  y = 0; y < src.rows; y++)
        minfilt(s_data+y*step, d_data+y*step, 1, src.cols, width);

    d_data = dst.ptr<uchar>(0);

    for(int  x = 0; x < src.cols; x++)
        minfilt(d_data+x, step, src.rows, width);

}

void Features::maxfilt(uchar* data, uchar* maxvalues, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    maxvalues[0] = data[0];
    for(unsigned int i=0; i < d-step; i+=step) {
        for(unsigned int k=i; k<d+i; k+=step) {
            if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
        }
        maxvalues[i+step] = maxvalues[i];
    }

    maxvalues[size-step] = data[size-step];
    for(unsigned int i=size-step; i > size-d; i-=step) {
        for(unsigned int k=i; k>i-d; k-=step) {
            if(data[k]>maxvalues[i]) maxvalues[i] = data[k];
        }
        maxvalues[i-step] = maxvalues[i];
    }

    deque<int> maxfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            maxvalues[i-d] = data[maxfifo.size()>0 ? maxfifo.front(): i-step];
        }

        if(data[i] < data[i-step]) {
            maxfifo.push_back(i-step);
            if(i==  width+maxfifo.front())
                maxfifo.pop_front();
        }
        else {
            while(maxfifo.size() > 0) {
                if(data[i] <= data[maxfifo.back()]) {
                    if(i==  width+maxfifo.front())
                    maxfifo.pop_front();
                    break;
                }
                maxfifo.pop_back();
            }
        }
    }

    maxvalues[size-d] = data[maxfifo.size()>0 ? maxfifo.front():size-step];

}

void Features::maxfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    deque<uchar> tmp;

    tmp.push_back(data[0]);
    for(unsigned int k=step; k<d; k+=step) {
        if(data[k]>tmp.back()) tmp.back() = data[k];
    }

    for(unsigned int i=step; i < d-step; i+=step) {
        tmp.push_back(tmp.back());
        if(data[i+d-step]>tmp.back()) tmp.back() = data[i+d-step];
    }


    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
            data[i-width] = tmp.front();
            tmp.pop_front();
        }

        if(data[i] < data[i-step]) {

            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
            minfifo.pop_front();

        } else {

            while(minfifo.size() > 0) {
                if(data[i] <= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                    minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }
        }
    }

    tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);

    for(unsigned int k=size-step-step; k>=size-d; k-=step) {
        if(data[k]>data[size-step]) data[size-step] = data[k];
    }

    for(unsigned int i=size-step-step; i >= size-d; i-=step) {
        data[i] = data[i+step];
        if(data[i-d+step]>data[i]) data[i] = data[i-d+step];
    }

    for(unsigned int i=size-width; i<=size-d; i+=step) {
        data[i] = tmp.front();
        tmp.pop_front();
    }

}


void Features::minfilt(uchar* data, uchar* minvalues, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    minvalues[0] = data[0];
    for(unsigned int i=0; i < d-step; i+=step) {
        for(unsigned int k=i; k<d+i; k+=step) {
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        minvalues[i+step] = minvalues[i];
    }

    minvalues[size-step] = data[size-step];
    for(unsigned int i=size-step; i > size-d; i-=step) {
        for(unsigned int k=i; k>i-d; k-=step) {
            if(data[k]<minvalues[i]) minvalues[i] = data[k];
        }
        minvalues[i-step] = minvalues[i];
    }

    deque<int> minfifo;

    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            minvalues[i-d] = data[minfifo.size()>0 ? minfifo.front(): i-step];
        }

        if(data[i] > data[i-step]) {
            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
            minfifo.pop_front();
        }

        else {
            while(minfifo.size() > 0) {
                if(data[i] >= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                        minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }
        }
    }

    minvalues[size-d] = data[minfifo.size()>0 ? minfifo.front():size-step];

}

void Features::minfilt(uchar* data, unsigned int step, unsigned int size, unsigned int width) {

    unsigned int d = int((width+1)/2)*step;
    size *= step;
    width *= step;

    deque<uchar> tmp;

    tmp.push_back(data[0]);
    for(unsigned int k=step; k<d; k+=step) {
        if(data[k]<tmp.back()) tmp.back() = data[k];
    }

    for(unsigned int i=step; i < d-step; i+=step) {
        tmp.push_back(tmp.back());
        if(data[i+d-step]<tmp.back()) tmp.back() = data[i+d-step];
    }

    deque<int> minfifo;
    for(unsigned int i = step; i < size; i+=step) {
        if(i >= width) {
            tmp.push_back(data[minfifo.size()>0 ? minfifo.front(): i-step]);
            data[i-width] = tmp.front();
            tmp.pop_front();
        }

        if(data[i] > data[i-step]) {
            minfifo.push_back(i-step);
            if(i==  width+minfifo.front())
            minfifo.pop_front();
        }

        else {
            while(minfifo.size() > 0) {
                if(data[i] >= data[minfifo.back()]) {
                    if(i==  width+minfifo.front())
                    minfifo.pop_front();
                    break;
                }
                minfifo.pop_back();
            }
        }
    }

    tmp.push_back(data[minfifo.size()>0 ? minfifo.front():size-step]);

    for(unsigned int k=size-step-step; k>=size-d; k-=step) {
        if(data[k]<data[size-step]) data[size-step] = data[k];
    }

    for(unsigned int i=size-step-step; i >= size-d; i-=step) {
        data[i] = data[i+step];
        if(data[i-d+step]<data[i]) data[i] = data[i-d+step];
    }

    for(unsigned int i=size-width; i<=size-d; i+=step) {
        data[i] = tmp.front();
        tmp.pop_front();
    }
}
