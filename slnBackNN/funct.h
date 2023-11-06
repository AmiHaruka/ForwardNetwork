#pragma once

#include <vector>
#include <string>
#include <cmath>

using namespace std;

//防止sigmoid命名冲突
namespace funct {
	static double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}
};
//vector<double>sigmoid();


//二分类softmax
vector<double> softmax(vector<double>);

//交叉熵
//double cross_entropy();