#ifndef _SVM_H_H_H
#define _SVM_H_H_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//#include <string>

#define _GLIBCXX_USE_CXX11_ABI 0

using namespace std;
using namespace cv;

 class CSVM
  {
  public:
    CSVM() {}
    ~CSVM() {}
    int m_nFeatNum; //图片中目标个数
	int m_nObjNum; //图片中最大检测到的目标个数
	CvSVM m_cSVM;
	int load_model(std::string &sPathModel);
	double predict(std::vector<double> vdConf[], int nNum);
   private:
	void splitString(string str, string sStandard, vector<string> &_vsStr);
	vector<string> split(string str, string pattern);

  };   

#endif