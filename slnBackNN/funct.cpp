#include "funct.h"
#include "layer.h"





//能够支持多个变量的接收的sigmoid
/*sigmoid(const vector<double>& values) {
	vector<double> results;
	results.reserve(values.size());

	for (const double& value : values) {
		results.push_back(1.0 / (1.0 + exp(-value)));
	}

	return results;
};
*/


//分类softmax
vector<double> softmax(vector<double> ipt) {
	vector<double> result;
	double total = 0.0;
	for (auto x : ipt) {
		total += exp(x);
	};
	for (auto x : ipt) {
		result.push_back(exp(x) / total);
	};
	return result;
};

//交叉熵废案
/*double cross_entropy(const vector<double>& predict, const vector<double>& actual) {
	double entropy = 0.f;
	for (size_t i = 0; i < actual.size(); i++) {
		entropy -= actual[i] * log(predict[i]);
	}
	//我不想平均化
	return entropy;
}
*/


//
