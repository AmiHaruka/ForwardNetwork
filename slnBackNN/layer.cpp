#pragma once

#include "layer.h"
#include "funct.h"
#include <random>
#include "process.h"

using namespace std;

Node::Node(size_t nextLayerSize) {
    weight.resize(nextLayerSize);
    del_weight.resize(nextLayerSize);
};

Layer::Layer() {
    //设置随机器
    mt19937 rd;
    rd.seed(random_device()());
    //随机函数
    uniform_real_distribution<double> distribution(-1, 1);

    //初始化输出层
    for (size_t i = 0; i < IgnNode; ++i) {

        inputLayer[i] = new Node(HidNode);

        for (size_t j = 0; j < HidNode; ++j) {

            inputLayer[i]->weight[j] = distribution(rd);

            inputLayer[i]->del_weight[j] = 0.f;
        }
    }

    //初始化隐藏层
    for (size_t j = 0; j < HidNode; ++j) {

        HidLayer[j] = new Node(OutNode);

        HidLayer[j]->bias = distribution(rd);

        HidLayer[j]->del_bias = 0.f;

        for (size_t k = 0; k < OutNode; ++k) {

            HidLayer[j]->weight[k] = distribution(rd);

            HidLayer[j]->del_weight[k] = 0.f;
        }
    }

    // 初始化输出层
    for (size_t k = 0; k < OutNode; ++k) {

        OutLayer[k] = new Node(0);

        OutLayer[k]->bias = distribution(rd);

        OutLayer[k]->del_bias = 0.f;
    }
};;



void Layer::initalize() {

    // 清理del_weight
    for (auto& nodeOfinputLayer : inputLayer) {
        nodeOfinputLayer->del_weight.assign(nodeOfinputLayer->del_weight.size(), 0.f);
    }

    // 清理del_weight
    for (auto& nodeOfHidLayer : HidLayer) {
        nodeOfHidLayer->del_bias = 0.f;
        nodeOfHidLayer->del_weight.assign(nodeOfHidLayer->del_weight.size(), 0.f);
    }

    // del_bias
    for (auto& nodeOfOutLayer : OutLayer) {
        nodeOfOutLayer->del_bias = 0.f;
    }
};

void Layer::forward() {

    /**
     * 输入层向中间层进行仿射变换
     * MathJax formula: h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
     */
    for (size_t j = 0; j < HidNode; ++j) {
        double sum = 0;
        for (size_t i = 0; i < IgnNode; ++i) {
            sum += inputLayer[i]->value * inputLayer[i]->weight[j];
        }
        sum -= HidLayer[j]->bias;

        HidLayer[j]->value = funct::sigmoid(sum);
    }

    /**
     * 中间层向输出层进行仿射变换
     * MathJax formula: \hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
     */
    for (size_t k = 0; k < OutNode; ++k) {
        double sum = 0;
        for (size_t j = 0; j < HidNode; ++j) {
            sum += HidLayer[j]->value * HidLayer[j]->weight[k];
        }
        sum -= OutLayer[k]->bias;

        OutLayer[k]->value = funct::sigmoid(sum);
    }

};

double Layer::computeLoss(const vector<double>& label) {
    double loss = 0.f;

    /**   废案
     * MathJax formula: Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
     */
    for (size_t k = 0; k < OutNode; ++k) {
        double tmp = std::fabs(OutLayer[k]->value - label[k]);
        loss += tmp * tmp / 2;
    }

    return loss;
};


//适用于MNIST手写数字分类任务的损失函数
double Layer::crossEntropyLoss(const vector<double>& predict, const vector<double>& actual) {
    double total_loss = 0.f;

    for (int i = 0; i < 10; ++i) {
        total_loss += -actual[i] * log(predict[i] + 1e-15);
    }
    return total_loss / 10;
}


//读取Node对象储存的value
vector<double> Layer::getOutputValues() {
    vector<double>  Values{};

    for (size_t k = 0; k < OutNode; ++k) {
        Values.push_back(OutLayer[k]->value);
    }

    return Values;
};

void Layer::backward(const vector<double>& label) {

    /*
     * MathJax formula: \Delta \lambda_k = - \eta (y_k - \hat{y_k}) \hat{y_k} (1 - \hat{y_k})
     */
    for (size_t k = 0; k < OutNode; ++k) {
        double del_bias =
            -(label[k] - OutLayer[k]->value)
            * OutLayer[k]->value * (1.0 - OutLayer[k]->value);

        OutLayer[k]->del_bias += del_bias;
    }

    /*
     * MathJax formula: \Delta v_{jk} = \eta ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) h_j
     */
    for (size_t j = 0; j < HidNode; ++j) {
        for (size_t k = 0; k < OutNode; ++k) {
            double del_weight =
                (label[k] - OutLayer[k]->value)
                * OutLayer[k]->value * (1.0 - OutLayer[k]->value)
                * HidLayer[j]->value;

            HidLayer[j]->del_weight[k] += del_weight;
        }
    }

    /*
     * MathJax formula: \Delta \beta_j = - \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j )
     */
    for (size_t j = 0; j < HidNode; ++j) {
        double del_bias = 0.f;
        for (size_t k = 0; k < OutNode; ++k) {
            del_bias +=
                -(label[k] - OutLayer[k]->value)
                * OutLayer[k]->value * (1.0 - OutLayer[k]->value)
                * HidLayer[j]->weight[k];
        }
        del_bias *=
            HidLayer[j]->value * (1.0 - HidLayer[j]->value);

        HidLayer[j]->del_bias += del_bias;
    }

    /*
     * MathJax formula: \Delta w_{ij} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j ) x_i
     */
    for (size_t i = 0; i < IgnNode; ++i) {
        for (size_t j = 0; j < HidNode; ++j) {
            double del_weight = 0.f;
            for (size_t k = 0; k < OutNode; ++k) {
                del_weight +=
                    (label[k] - OutLayer[k]->value)
                    * OutLayer[k]->value * (1.0 - OutLayer[k]->value)
                    * HidLayer[j]->weight[k];
            }
            del_weight *=
                HidLayer[j]->value * (1.0 - HidLayer[j]->value)
                * inputLayer[i]->value;

            inputLayer[i]->del_weight[j] += del_weight;
        }
    }
};

/* 废案，当时思考变量跨文件共享时没想到extern关键字
*void Layer::setTrainLabels(const vector<vector<double>>& labels) {
*trainLabels = labels;  
}*/

bool Layer::train(const vector < vector<double> >& trainData) {

    cout << "Start Training" << endl;
    double total_loss = 0.f;
    for (size_t index = 0; index < 300; ++index) {

        initalize();

        for (size_t i = 0; i < 784; ++i) {
            inputLayer[i]->value = trainData[index][i];
        }
            forward();

            double loss = crossEntropyLoss(getOutputValues(), train_labels_onehot[index]);
            total_loss += loss;

            backward(train_labels_onehot[index]);
    }
    double avg_loss = total_loss / 300;
    printf("Training SUCCESS in 300 instance.\n");
    return true;
}

//适用于测试集
void Layer::predict(const vector<vector<double>>& TestData, const vector<vector<double>>& TrueLabels) {
    double pre_total_loss = 0.f;
    for (size_t i = 0; i < 100; ++i) {
        initalize();
        for (size_t j = 0; j < 784;++j) {
            inputLayer[j]->value = TestData[i][j];
        }
        forward();
        double loss = crossEntropyLoss(getOutputValues(), TrueLabels[i]);
        pre_total_loss += loss;
        cout << "Sample " << i << " - Loss: " << loss << endl;
    }
    double avg_loss = pre_total_loss / 100;
    cout << "Average Loss: " << avg_loss << endl;
}
