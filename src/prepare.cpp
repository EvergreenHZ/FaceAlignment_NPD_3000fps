#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "lbf/common.hpp"

using namespace cv;
using namespace std;
using namespace lbf;

CascadeClassifier cc("../model/haarcascade_frontalface_alt.xml");  //use opencv classifier

Rect getBBox(Mat &img, Mat_<double> &shape) {
        vector<Rect> rects;
        cc.detectMultiScale(img, rects, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));  // may misdetect
        if (rects.size() == 0) return Rect(-1, -1, -1, -1);  // pass
        double center_x, center_y, x_min, x_max, y_min, y_max;
        center_x = center_y = 0;
        x_min = x_max = shape(0, 0);
        y_min = y_max = shape(0, 1);
        for (int i = 0; i < shape.rows; i++) {
                center_x += shape(i, 0);
                center_y += shape(i, 1);
                x_min = min(x_min, shape(i, 0));
                x_max = max(x_max, shape(i, 0));
                y_min = min(y_min, shape(i, 1));
                y_max = max(y_max, shape(i, 1));
        }
        center_x /= shape.rows;
        center_y /= shape.rows;

        for (int i = 0; i < rects.size(); i++) {
                Rect r = rects[i];
                if (x_max - x_min > r.width*1.5) continue;
                if (y_max - y_min > r.height*1.5) continue;
                if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
                if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
                return r;
        }
        return Rect(-1, -1, -1, -1);
}

void genTxt(const string &inTxt, const string &outTxt) {  //inTxt: all the image names
        Config &config = Config::GetInstance();
        int landmark_n = config.landmark_n;
        Mat_<double> gt_shape(landmark_n, 2);  //68 x 2

        FILE *inFile = fopen(inTxt.c_str(), "r");
        FILE *outFile = fopen(outTxt.c_str(), "w");
        assert(inFile && outFile);

        char line[256];
        char buff[1000];
        string out_string("");
        int N = 0;
        while (fgets(line, sizeof(line), inFile)) {  //for each images
                string img_path(line, strlen(line) - 1);

                LOG("Handle %s", img_path.c_str());

                string pts = img_path.substr(0, img_path.find_last_of(".")) + ".pts";  //find description file
                FILE *tmp = fopen(pts.c_str(), "r");
                assert(tmp);
                //kick out of the first 3 lines
                fgets(line, sizeof(line), tmp);  //read line from tmp file to line
                fgets(line, sizeof(line), tmp);
                fgets(line, sizeof(line), tmp);
                for (int i = 0; i < landmark_n; i++) {   //68, read all the landmarks
                        fscanf(tmp, "%lf", &gt_shape(i, 0));  //x coordinate
                        fscanf(tmp, "%lf", &gt_shape(i, 1));  //y ...
                }
                fclose(tmp);

                Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
                Rect bbox = getBBox(img, gt_shape);  //get the bounding box

                if (bbox.x != -1) {
                        N++;
                        sprintf(buff, "%s %d %d %d %d", img_path.c_str(), bbox.x, bbox.y, bbox.width, bbox.height);
                        out_string += buff;  //concatenate
                        for (int i = 0; i < landmark_n; i++) {
                                sprintf(buff, " %lf %lf", gt_shape(i, 0), gt_shape(i, 1));
                                out_string += buff;
                        }
                        out_string += "\n";  // here, newline
                }
        }

        fprintf(outFile, "%d\n%s", N, out_string.c_str());  //good enough

        fclose(inFile);
        fclose(outFile);
}

int prepare(void) {
        Config &params = Config::GetInstance();
        string txt = params.dataset + "/Path_Images_train.txt";
        genTxt(txt, params.dataset + "/train.txt");
        txt = params.dataset + "/Path_Images_test.txt";
        genTxt(txt, params.dataset + "/test.txt");
        return 0;
}
