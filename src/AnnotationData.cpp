/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "AnnotationData.h"

#include <libconfig.h++>
#include <iostream>

using namespace libconfig;
using namespace std;


bool AnnotationData::loadAnnoFile(const char* filename) {

    Config configFile;

    cout<<filename<<endl;

    // try to read configuration file
    try {
    configFile.readFile(filename);
    }
    catch(const FileIOException &fioex) {
    cerr << "Could not read config file " << filename << endl;
    return false;
    }
    catch(const ParseException &pex) {
    cerr << "Parse error: " << filename << endl;
    return false;
    }

    // look up parameter settings / values
    try{

        const Setting &image_files = configFile.getRoot()["image_files"];

        AnnoData.resize(image_files.getLength());
        for(unsigned int i=0;i<AnnoData.size();++i) {

            Annotation& anno       = AnnoData[i];
            anno.image_name        = (const char*)image_files[i]["filename"];
            anno.sample_background = image_files[i]["sample_background"];
            anno.hard_neg          = image_files[i]["hard_neg"];

            const Setting &pos = image_files[i]["bb_pos"];
            anno.posROI.resize(pos.getLength());
            for(int unsigned j=0;j<anno.posROI.size();++j) {
                anno.posROI[j].x      = pos[j][0];
                anno.posROI[j].y      = pos[j][1];
                anno.posROI[j].width  = pos[j][2];
                anno.posROI[j].height = pos[j][3];
            }

            const Setting &neg = image_files[i]["bb_neg"];
            anno.negROI.resize(neg.getLength());
            for(int unsigned j=0; j<anno.negROI.size(); ++j) {
                anno.negROI[j].x      = neg[j][0];
                anno.negROI[j].y      = neg[j][1];
                anno.negROI[j].width  = neg[j][2];
                anno.negROI[j].height = neg[j][3];
            }
        }
    }

    catch(const SettingNotFoundException &nfex) {
        cerr << "Not found in configuration file!" << endl;
        return false;
    }
    return true;
}

void AnnotationData::reorder() {

    vector<Annotation> data(AnnoData);
    AnnoData.clear();

    int          denom = 2;
    unsigned int count = 1;
    while(count<data.size()) {
        for(int j=denom-1;j>0;j-=2) {
            int index = int( (j*data.size())/denom );
            AnnoData.push_back(data[index]);
            data.erase(data.begin()+index);
        }
        count = denom;
        denom *= 2;
    }

    for(unsigned int i=0;i<data.size();++i) {
        AnnoData.push_back(data[i]);
    }

}

void AnnotationData::getStatistics(StructParam* par, int i1, int i2) {
    vector<int> width;
    vector<int> height;

    if(i2==-1) i2 = AnnoData.size();

    for(int i=i1;i<i2;++i) {
        for(unsigned int j=0;j<AnnoData[i].posROI.size(); ++j) {
            width.push_back(AnnoData[i].posROI[j].width);
            height.push_back(AnnoData[i].posROI[j].height);
//            cout<<(float)AnnoData[i].posROI[j].height/(float)AnnoData[i].posROI[j].width<<endl;
        }
    }

    // median
    sort(height.begin(),height.end());
    sort(width.begin(),width.end());
    int w = width[int(width.size()/2)];
    int h = height[int(height.size()/2)];

    double u_size;
    vector<int>* vsize;

    // limit units to be less than 128
    if(w>h) {
        par->u_width = w*par->u_scale;
        par->u_height = h*par->u_scale;

        u_size = par->u_width;
        vsize = &width;
    }
    else {
        par->u_width = w*par->u_scale;
        par->u_height = h*par->u_scale;

        u_size = par->u_height;
        vsize = &height;
    }

    // scales
    double min_scale = u_size/double(vsize->back());
    double max_scale = u_size/double(vsize->front());

    cout << "Unit Size: " << par->u_width << " " << par->u_height << endl;
    computeScale(par, min_scale, max_scale);

}

bool AnnotationData::computeScale(StructParam* par, double min_scale, double max_scale) {

    unsigned int old_size = par->scales.size();

    par->scales.clear();
    double d = (par->u_scale - min_scale - par->d_scale/2.0);
    if(d>0) {
        int num_scales = (int)ceil(d/par->d_scale);
        // minimum scale >0
        if(par->d_scale*num_scales >= par->u_scale) --num_scales;

        for(int i=num_scales; i>0; --i)
        par->scales.push_back(par->u_scale - i*par->d_scale);
    }
    par->scales.push_back(par->u_scale);

    d = (max_scale-par->u_scale-par->d_scale/2.0);
    if(d>0) {
        int num_scales = (int)ceil(d/par->d_scale);
        for(int i=1; i<=num_scales; ++i)
        par->scales.push_back(par->u_scale + i*par->d_scale);
    }

    par->inv_scales.resize(par->scales.size());
    for (unsigned int i = 0;i<par->scales.size();i++) {
        cout << par->scales[i] << " ";
        par->inv_scales[i] = 1.0/par->scales[i];
    }
    cout << endl;

    return (old_size!=par->scales.size());
}
