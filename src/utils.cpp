#include "lbf/lbf.hpp"

#include <cstdio>
#include <stdio.h>
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include <vector>
#include "npd/npddetect.h"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace lbf;
using namespace std;

void prepareImages(string outfile, vector< vector<Mat> > & img_vec)
{
        string dir;
        ifstream outer_file(outfile.c_str());
        while(getline(outer_file, dir)) {  // for each identity person, in a directory
                ifstream inner_file( dir.c_str());
                string img_path;
                vector<Mat> imgs;
                while(getline(inner_file,img_path)) {  // read all the images of the same person
                        imgs.push_back(imread((dir + img_path).c_str(), 0));  // get the correct path and read image
                }
                img_vec.push_back(imgs);
        }
        return ;
}

void getFaceVec(vector< vector<Mat> > & img_vec, vector< vector<Mat> > & face_vec, vector< vector<BBox> >& bbox_vec, string detection_model_path)
{

        /* start detection */
        npd::npddetect npd;
        npd.load(detection_model_path.c_str());  //load

        for (int i = 0; i < img_vec.size(); i++) {
                float score = 15.;
                vector<Mat> faces;
                vector<BBox> bboxes;
                printf("************* [TEST] Npd:detect test... *************\n");
                for (int j = 0; j < img_vec[i].size(); j++) {
                        Mat img = img_vec[i][j];  // get image

                        int n;
                        double t = (double)cvGetTickCount();
                        n = npd.detect(img.data, img.cols, img.rows);  // actually, there is only one face
                        t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;

                        printf("Detect num: %d (%lf ms avg of %d test)\n", n, t/n, n);
                        if (n == 0) {
                                continue;
                        }

                        vector< int >& Xs = npd.getXs();
                        vector< int >& Ys = npd.getYs();
                        vector< int >& Ss = npd.getSs();
                        vector< float >& Scores = npd.getScores();

                        // crop img
                        int index = -1;
                        for (int k = 0; k < Scores.size(); k++) {
                                if (Scores[k] < score) {
                                        continue;
                                }else{
                                        index = k;  // just consider one face founded
                                        break;
                                }
                        }

                        if (index == -1) continue;

                        int x = Xs[index], y = Ys[index], w = Ss[index], h = Ss[index], half_size = w / 2;
                        int x_min = 0, x_max = 0, y_min = 0, y_max = 0;
                        x_min = max(0., (double)x - half_size);
                        x_max = min(img.cols - 1., x + 3. * half_size);
                        y_min = max(0., (double)y - half_size);
                        y_max = min(img.rows - 1., y + 3. * half_size);

                        BBox bbox_(x - x_min, y - y_min, w, h);  // what you need is just the face box(roi)
                        bboxes.push_back(bbox_);
                        Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
                        img = img(roi).clone();

                        faces.push_back(img);
                }

                if(!faces.size()) continue;
                face_vec.push_back(faces);
                bbox_vec.push_back(bboxes);
        }
        img_vec.clear();  // now I don't need imgs, faces and bboxes are enough
}

void getLandmarks(vector< vector<Mat> > & face_vec, vector< vector<BBox> > &bbox_vec, vector< vector<Mat> > & shape_vec)
{

        Config &config = Config::GetInstance();
        int landmark_n = config.landmark_n;

        /* load the model */
        LbfCascador lbf_cascador;
        FILE *model = fopen(config.saved_file_name.c_str(), "rb");  // model path
        lbf_cascador.Read(model);
        fclose(model);


        for (int i = 0; i < face_vec.size(); i++) {
                vector<Mat> shapes;
                for (int j = 0; j < face_vec[i].size(); j++) {
                        cout<<"Execute alignment"<<endl;
                        Mat shape = lbf_cascador.Predict(face_vec[i][j], bbox_vec[i][j]);
                        shapes.push_back(shape);
                }
                shape_vec.push_back(shapes);
        }

        return ;
}

void getFeatureVec(vector< vector<Mat> > & face_vec, vector< vector<Mat> > & shape_vec, vector< vector<Mat_<int> > > & fea_vec)
{
        for (int i = 0; i < face_vec.size(); i++) {
                vector<Mat_<int> > t_fea;
                for (int j = 0; j < face_vec[i].size(); j++) {
                        Mat_<int> fea(shape_vec[0][0].rows, 1);  // may be not usigned char;
                        for (int k = 0; k < shape_vec[i][j].rows; k++) {
                                fea.at<int>(k, 0) = 
                                        (int)face_vec[i][j].at<unsigned char>(
                                                        int(shape_vec[i][j].at<unsigned char>(k, 0)),
                                                        int(shape_vec[i][j].at<unsigned char>(k, 1)));  // for a specific face, get the feature vector.
                        }
                        t_fea.push_back(fea);
                }
                fea_vec.push_back(t_fea);
        }

        return ;
}

void getPosSamle(vector< vector<Mat_<int> > > & fea_vec, vector< Mat_<int> > & pos_sample)
{
        for (int i = 0; i < fea_vec.size(); i++) {  // for each same identity person
                vector<Mat_<int> > idx_landmark_pixels = fea_vec[i];
                for (int j = 0; j < idx_landmark_pixels.size(); j++) {
                        Mat_<int> fixed_one = idx_landmark_pixels[j];
                        for (int k = 0; k < idx_landmark_pixels.size(); k++) {
                                Mat_<int> walked_one = idx_landmark_pixels[k];
                                pos_sample.push_back(fixed_one - walked_one);
                        }
                }
        }
        return ;
}

void getNegSample(vector< vector<Mat_<int> > > & fea_vec, vector< Mat_<int> > & neg_sample)
{

        for (int i = 0; i < fea_vec.size(); i++) {  // for person x
                vector<Mat_<int> > idx_landmark_pixels = fea_vec[i];
                for (int j = i + 1; j < fea_vec.size(); j++) {  // for person y
                        vector<Mat_<int> > idy_landmark_pixels = fea_vec[j];

                        for (int u = 0; u < idx_landmark_pixels.size(); u++) {  // calculate the difference vector
                                Mat_<int> idx = idx_landmark_pixels[u];
                                for (int v = 0; v < idy_landmark_pixels.size(); v++) {
                                        Mat_<int> idy = idy_landmark_pixels[v];

                                        neg_sample.push_back(idx - idy);
                                }
                        }
                }
        }
        return ;
}
