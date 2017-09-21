/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HTTPJSONAPI_H
#define HTTPJSONAPI_H

#include "jsonapi.h"
#include <math.h>
#include <boost/network/protocol/http/server.hpp>
#include <boost/network/uri.hpp>
#include <boost/network/uri/uri_io.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

//#include <fstream>
#include <unistd.h>
#include <algorithm>

//#define _GLIBCXX_USE_CXX11_ABI 0

namespace http = boost::network::http;
namespace uri = boost::network::uri;
class APIHandler;
typedef http::server<APIHandler> http_server;

namespace dd
{
  std::string uri_query_to_json(const std::string &req_query);
  
  class HttpJsonAPI : public JsonAPI
  {
  public:
    HttpJsonAPI();
    ~HttpJsonAPI();

    void stop_server();
    int start_server_daemon(const std::string &host,
			    const std::string &port,
			    const int &nthreads);
    int start_server(const std::string &host,
		     const std::string &port,
		     const int &nthreads);
    int boot(int argc, char *argv[]);
    static void terminate(int param);
    
    http_server *_dd_server = nullptr; /**< main reusable pointer to server object */
    std::future<int> _ft; /**< holds the results from the main server thread */
  };
  class CSVM
  {
  public:
    CSVM() {}
    ~CSVM() {}
    int m_nFeatNum; //图片中目标个数
	int m_nObjNum; //图片中最大检测到的目标个数
	CvSVM m_cSVM;
	int load_model(std::string &sPathModel){
    m_nFeatNum = 10; //图片中目标个数
	m_nObjNum = 10; //图片中最大检测到的目标个数	
	
    std::string sTemp = sPathModel + "_param";
	if(access(sPathModel.c_str(),0)==-1 || access(sTemp.c_str(),0)==-1)
	{
		//cout<<"The file does not exist!:"<<sPathModel<<" or "<<sTemp<<endl;
		return -1;
	}
	
	m_cSVM.load(sPathModel.c_str());
	std::fstream file;
	file.open(sTemp.c_str(),std::ios::in);
	char buffer[256];
	std::string str;
	while(!file.eof())
	{
		file.getline(buffer,256,'\n');//getline(char *,int,char) 表示该行字符达到256个或遇到换行就结束
		sTemp = buffer;
		std::vector<std::string> vs;
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
   };
	double predict(std::vector<double> vdConf[], int nNum){
        //转换数据	
	int nWidth = m_nObjNum*m_nFeatNum;
	cv::Mat sample = cv::Mat::zeros(1, nWidth, CV_32FC1);
	int nCol = 0;
	int n = 0;
    nNum = cv::min(nNum, m_nFeatNum);
	for (int i=0; i<nNum; i++)
	{
		n = vdConf[i].size();
        n = cv::min(n, m_nObjNum);
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
	double dCof = 1.0- cv::max(cv::min((float)(1 / (1 + exp((-1.0)*dSum))),float(1.0)),float(0.0));
		
	return dCof;
    };

	void splitString(std::string str, std::string sStandard, std::vector<std::string> &_vsStr){
    _vsStr.clear();
	std::vector<std::string> vs = split(str, sStandard);
	//cout<<"yuan = "<<str<<endl;
	for (int i=0; i<vs.size(); i++)
	{
		_vsStr.push_back(vs[i]);
		//cout<<vs[i]<<endl;
	}
    };
	std::vector<std::string> split(std::string str, std::string pattern){
        std::string::size_type pos;
	std::vector<std::string> result;
	str+=pattern;//扩展字符串以方便操作
	int size=str.size();
 
	for(int i=0; i<size; i++)
	{
		pos=str.find(pattern,i);
		if(pos<size)
		{
			std::string s=str.substr(i,pos-i);
			result.push_back(s);
			i=pos+pattern.size()-1;
		}
	}
	return result;
    };
  };   
}

#endif
