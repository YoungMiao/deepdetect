#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include "svm_own.h"
#include <fstream>
#include <unistd.h>
#include <math.h>
#include <algorithm>

using namespace cv;
using namespace std;


CSVM::CSVM(void)
{
	
}

CSVM::CSVM(int nFeatNum, int nObjNum)
{
	m_nFeatNum = nFeatNum; //图片中目标个数
	m_nObjNum = nObjNum; //图片中最大检测到的目标个数	
}

int CSVM::load_model()//string sPathModel)
{
    m_nFeatNum = 10; //图片中目标个数
	m_nObjNum = 10; //图片中最大检测到的目标个数	
	
    string sTemp = sPathModel + "_param";
	if(access(sPathModel.c_str(),0)==-1 || access(sTemp.c_str(),0)==-1)
	{
		cout<<"The file does not exist!:"<<sPathModel<<" or "<<sTemp<<endl;
		return -1;
	}
	
	m_cSVM.load(sPathModel.c_str());
	fstream file;
	file.open(sTemp.c_str(),ios::in);
	char buffer[256];
	string str;
	while(!file.eof())
	{
		file.getline(buffer,256,'\n');//getline(char *,int,char) 表示该行字符达到256个或遇到换行就结束
		sTemp = buffer;
		vector<string> vs;
		splitString(sTemp, "=", vs);
		if (2==vs.size())
		{
			if (vs[0] == "feat_num")
				m_nFeatNum = atoi(vs[1].c_str());
			if (vs[0] == "obj_num")
				m_nObjNum = atoi(vs[1].c_str());
		}
	}
	return 0;
}

double CSVM::predict(vector<double> vdConf[], int nNum)
{
	//转换数据	
	int nWidth = m_nObjNum*m_nFeatNum;
	Mat sample = Mat::zeros(1, nWidth, CV_32FC1);
	int nCol = 0;
	int n = 0;
    nNum = min(nNum, m_nFeatNum);
	for (int i=0; i<nNum; i++)
	{
		n = vdConf[i].size();
        n = min(n, m_nObjNum);
		for (int j=0; j<n; j++)
		{
			nCol = i*m_nObjNum + j;
			sample.at<float>(0,nCol) = vdConf[i][j];
			//cout<<vdConf[i][j]<<endl;
		}
	}
	
	//预测
	int nPredictLabel = 0;
	double dSum = m_cSVM.predict(sample,true);
	double dCof = 1.0- max(min((float)(1 / (1 + exp((-1.0)*dSum))),float(1.0)),float(0.0));
		
	return dCof;
}

void CSVM::splitString(string str, string sStandard, vector<string> &_vsStr)
{
	_vsStr.clear();
	vector<string> vs = split(str, sStandard);
	//cout<<"yuan = "<<str<<endl;
	for (int i=0; i<vs.size(); i++)
	{
		_vsStr.push_back(vs[i]);
		//cout<<vs[i]<<endl;
	}
}

//字符串分割函数
vector<string> CSVM::split(string str, string pattern)
{
	string::size_type pos;
	vector<string> result;
	str+=pattern;//扩展字符串以方便操作
	int size=str.size();
 
	for(int i=0; i<size; i++)
	{
		pos=str.find(pattern,i);
		if(pos<size)
		{
			string s=str.substr(i,pos-i);
			result.push_back(s);
			i=pos+pattern.size()-1;
		}
	}
	return result;
}
