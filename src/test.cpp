#include "lbf/lbf.hpp"
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <vector>
#include "npd/npddetect.h"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace lbf;

// dirty but works
void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

void getLandmarkVector(Mat& face, Mat& shape, vector<unsigned char> &fea_vec)
{
        /*for (int i = 0; i < shape.rows; i++) {
          fea_vec.push_back(
          face.( shape(i, 0), shape(i, 1)));
          }*/
        return ;
}

int test(void) {
        Config &config = Config::GetInstance();

        LbfCascador lbf_cascador;
        FILE *fd = fopen(config.saved_file_name.c_str(), "rb");
        lbf_cascador.Read(fd);
        fclose(fd);

        LOG("Load test data from %s", config.dataset.c_str());
        string txt = config.dataset + "/test.txt";
        vector<Mat> imgs, gt_shapes;
        vector<BBox> bboxes;
        parseTxt(txt, imgs, gt_shapes, bboxes);

        int N = imgs.size();
        lbf_cascador.Test(imgs, gt_shapes, bboxes);

        return 0;
}

int run(void) {
        Config &config = Config::GetInstance();
        FILE *fd = fopen((config.dataset + "/test.txt").c_str(), "r");
        assert(fd);
        int N;
        int landmark_n = config.landmark_n;
        fscanf(fd, "%d", &N); // N = 315 for test
        char img_path[256];
        double bbox[4];
        vector<double> x(landmark_n), y(landmark_n);

        LbfCascador lbf_cascador;
        FILE *model = fopen(config.saved_file_name.c_str(), "rb");  // model file
        lbf_cascador.Read(model);
        fclose(model);

        for (int i = 0; i < N; i++) {  // for each image
                fscanf(fd, "%s", img_path);
                for (int j = 0; j < 4; j++) {
                        fscanf(fd, "%lf", &bbox[j]);  // get the box coordinates
                }
                for (int j = 0; j < landmark_n; j++) {
                        fscanf(fd, "%lf%lf", &x[j], &y[j]);
                }
                Mat img = imread(img_path);
                // crop img
                double x_min, y_min, x_max, y_max;
                x_min = *min_element(x.begin(), x.end());
                x_max = *max_element(x.begin(), x.end());
                y_min = *min_element(y.begin(), y.end());
                y_max = *max_element(y.begin(), y.end());

                x_min = max(0., x_min - bbox[2] / 2);
                x_max = min(img.cols - 1., x_max + bbox[2] / 2);
                y_min = max(0., y_min - bbox[3] / 2);
                y_max = min(img.rows - 1., y_max + bbox[3] / 2);

                double x_, y_, w_, h_;
                x_ = x_min; y_ = y_min;
                w_ = x_max - x_min; h_ = y_max - y_min;
                BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);  // what you need is just the face box(roi)
                Rect roi(x_, y_, w_, h_);
                img = img(roi).clone();

                Mat gray;
                cvtColor(img, gray, CV_BGR2GRAY);
                LOG("Run %s", img_path);
                Mat shape = lbf_cascador.Predict(gray, bbox_);
                img = drawShapeInImage(img, shape, bbox_);
                imshow("landmark", img);
                waitKey(0);
        }
        fclose(fd);
        return 0;
}

int align(char* imgName)
{
        /* start detection */
        //printf("************* [TEST] Npd:detect test... *************\n");

        npd::npddetect npd;
        npd.load("../model/result.bin");  //load
        cv::Mat color = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);
        cv::Mat img;
        cvtColor(color, img, CV_BGR2GRAY);

        int nt = 1; 
        int nc = nt; 
        int n;
        double t = (double)cvGetTickCount();
        while(nc-- > 0)
                n = npd.detect(img.data, img.cols, img.rows);
        t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;
        printf("%s, %4d, %4d, %4d, %lf\n", imgName, img.rows, img.cols, img.rows * img.cols,  t);

        //printf("Detect num: %d (%lf ms avg of %d test)\n", n, t/nt, nt);
        vector< int >& Xs = npd.getXs();
        vector< int >& Ys = npd.getYs();
        vector< int >& Ss = npd.getSs();
        vector< float >& Scores = npd.getScores();

        //printf("************* [TEST] Npd:detect ok!!!!! *************\n\n");

        if (n == 0) {
                string save_name("npd_failed_shot/");
                save_name += imgName;
                imwrite(save_name, color);
                //printf("No Face Founded, Process Terminated!\n\n");
                return 0;
                //exit(0);
        } else {
                string saved_name("npd_successful_shot/");
                saved_name += imgName;
                imwrite(saved_name, color);

        }


        /* start alignment */
        Config &config = Config::GetInstance();
        int landmark_n = config.landmark_n;

        /* load the model */
        LbfCascador lbf_cascador;
        FILE *model = fopen(config.saved_file_name.c_str(), "rb");
        lbf_cascador.Read(model);
        fclose(model);

        // crop img
        int index = 0;
        float score = 15.;
        for (int i = 0; i < Scores.size(); i++) {
                if (Scores[i] < score) {
                        continue;
                }else{
                        index = i;
                        break;
                }
        }

        int x = Xs[index], y = Ys[index], w = Ss[index], h = Ss[index], half_size = w / 2;
        int x_min = 0, x_max = 0, y_min = 0, y_max = 0;
        x_min = max(0., (double)x - half_size);
        x_max = min(img.cols - 1., x + 3. * half_size);
        y_min = max(0., (double)y - half_size);
        y_max = min(img.rows - 1., y + 3. * half_size);

        BBox bbox_(x - x_min, y - y_min, w, h);  // what you need is just the face box(roi)
        Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
        img = img(roi).clone();
        Mat show_color = color(roi).clone();

        //LOG("Run %s", imgName);
        Mat shape = lbf_cascador.Predict(img, bbox_);

        //vector<unsigned char> fea_vec;
        //getLandmarkVector(img, shape, fea_vec);
        //img = drawShapeInImage(img, shape, bbox_);
        //imshow("landmark", img);
        show_color = drawShapeInImage(show_color, shape, bbox_);
        //imshow("landmark", show_color);
        //waitKey(0);
        string saved_name("3000fps_successful_align/");
        saved_name += imgName;
        imwrite(saved_name, show_color);
        return 0;
}

// pass 灰度
//vector<vector<int> > detect_and_align(const unsigned char* I, int width, int height, const char* npd_model, char* align_model)
vector<int> detect_and_align(const unsigned char* I, int width, int height, const char* npd_model)
{
        // reconstruct the origin Mat
        Mat img(height, width, CV_8UC1, (void*)I);

        vector<int> point_vector;
        npd::npddetect npd;
        npd.load(npd_model);  //load

        int n;
        n = npd.detect(I, width, height);
        if (n == 0) {
                return point_vector;
        }

        vector< int >& Xs = npd.getXs();
        vector< int >& Ys = npd.getYs();
        vector< int >& Ss = npd.getSs();
        vector< float >& Scores = npd.getScores();

        /* start alignment */
        Config &config = Config::GetInstance();  // change model here
        int landmark_n = config.landmark_n;

        /* load the model */
        LbfCascador lbf_cascador;
        FILE *model = fopen(config.saved_file_name.c_str(), "rb");
        lbf_cascador.Read(model);
        fclose(model);

        // crop img
        int index = 0;
        float score = 15.;
        for (int i = 0; i < Scores.size(); i++) {
                if (Scores[i] < score) {
                        continue;
                }else{
                        index = i;
                        break;
                }
        }

        int x = Xs[index], y = Ys[index], w = Ss[index], h = Ss[index], half_size = w / 2;
        int x_min = 0, x_max = 0, y_min = 0, y_max = 0;
        x_min = max(0., (double)x - half_size);
        x_max = min(height - 1., x + 3. * half_size);
        y_min = max(0., (double)y - half_size);
        y_max = min(width - 1., y + 3. * half_size);

        //Mat ori = img.clone();

        BBox bbox_(x - x_min, y - y_min, w, h);  // what you need is just the face box(roi)
        Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);
        img = img(roi).clone();

        point_vector.push_back(x);
        point_vector.push_back(y);
        point_vector.push_back(w);
        point_vector.push_back(h);

        Mat shape = lbf_cascador.Predict(img, bbox_);
        //Mat ori = img.clone();
        img = drawShapeInImage(img, shape, bbox_);
        //namedWindow("win", WINDOW_AUTOSIZE);
        //imshow("win", img);
        //waitKey(0);

        //BBox bbox_ori(x, y, w, h);
        //for (int i = 0; i < shape.rows; i++) {
        //        shape.at<double>(i, 0) += x_min;
        //        shape.at<double>(i, 1) += y_min;
        //}
        //ori = drawShapeInImage(ori, shape, bbox_ori);
        //namedWindow("win", WINDOW_AUTOSIZE);
        //imshow("win", ori);
        //waitKey(0);



        for (int i = 0; i < shape.rows; i++) {
                point_vector.push_back(static_cast<int>(shape.at<double>(i, 0)) + x_min);
                point_vector.push_back(static_cast<int>(shape.at<double>(i, 1)) + y_min);
        }

        for (int i = 0; i < shape.rows; i++) {
                for (int j = 0; j < shape.cols; j++) {
                        cout << point_vector[i * shape.cols + j] << " ";
                }
                cout << endl;
        }

        cout << "rows: " << shape.rows << endl;
        cout << "cols: " << shape.cols << endl;


        return point_vector;
}
