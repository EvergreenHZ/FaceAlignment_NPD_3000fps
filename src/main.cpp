#include <cstdio>
#include "lbf/common.hpp"
#include <iostream>
#include <fstream>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>

using namespace std;
using namespace lbf;
using namespace cv;

// dirty but works
int train(int);
int test(void);
int prepare(void);
int run(void);
int align(char*);
vector<int> detect_and_align(const unsigned char* I, int width, int height, const char* npd_model);


int main(int argc, char **argv)
{
        if (strcmp(argv[1], "train") == 0) {  //now I just care about this
                return train(0);
        }

        if (strcmp(argv[1], "prepare") == 0) {
                return prepare();
        }
        if (strcmp(argv[1], "run") == 0) {
                return run();
        }
        if (strcmp(argv[1], "align") == 0) {
                return align(argv[2]);
        }
        if (strcmp(argv[1], "test") == 0) {
                ifstream infile(argv[2]);
                string name;
                while (getline(infile, name)) {
                        align(const_cast<char*>(name.c_str()));
                }
        }

        if (strcmp(argv[1], "hello") == 0) {
                Mat gray_imge = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
                detect_and_align(gray_imge.data, gray_imge.cols, gray_imge.rows, "../model/result.bin");
        }

        else {
                LOG("Wrong Arguments.");
        }
        return 0;
}
