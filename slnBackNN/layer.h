#pragma once

#include <vector>

using namespace std;

//输入，隐藏，输出
const size_t IgnNode = 784;
const size_t HidNode = 50;
const size_t OutNode = 10;

//学习率与一次迭代所用样本数量
//为了方便，这里我只实现300次训练，1次取100样本
const double lr = 0.05;
const int batch_size = 300;

//const int numEpochs = 300;  mini-batch的废案 

const size_t train_max_epoch = 600;
const size_t test_max_epoch = 100;


//节点，储存权重，偏置以及值
struct Node {
	double value{};
	double bias{};
	double del_bias{};
	vector<double> weight;
	vector<double> del_weight;


	explicit Node(size_t nextLayerSize);
};


//全连接层
class Layer {
private:
	Node* inputLayer[IgnNode]{};
	Node* HidLayer[HidNode]{};
	Node* OutLayer[OutNode]{};

	void initalize();
	void forward();
	double computeLoss(const vector<double>& label);
	double crossEntropyLoss(const vector<double>& predictions, const vector<double>& actualLabels);
	void backward(const vector<double>& label);

	// vector<vector<double>> trainLabels;  // 用于存储训练标签的废案

public:

	Layer();


	void setTrainLabels(const vector<vector<double>>&);
	void predict(const vector< vector<double> >&, const vector<vector<double>>&);
	vector<double> getOutputValues();
	bool train(const vector < vector<double> >& );


};
