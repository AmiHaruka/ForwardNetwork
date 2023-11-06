#pragma once

#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <direct.h>
#else
#include <unistd.h>
#endif

#include "process.h"
#include "layer.h"
#include <iostream>

using namespace std;

vector< vector<double> > train_images;
vector<double> train_labels;
vector< vector<double> > test_images;
vector<double> test_labels;


//以下实现了MNIST数据集的处理


//适应x86架构的小端字节
int reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
};



void read_train_images() {
	ifstream file("train-images.idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);

		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;
				};
			};
			train_images.push_back(this_image);
		};
		printf("%d, train images success\n", train_images.size());
	}
	else {
		printf("\nCan not found!");
	}
};



void read_train_labels() {
	ifstream file;
	file.open("train-labels.idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			train_labels.push_back((double)label);
		}
		printf("%d, train labels success\n", train_labels.size());
	}
}



void read_test_images() {
	ifstream file("t10k-images.idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);

		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;
				}
			}
			test_images.push_back(this_image);
		}
		printf("%d, test images success\n", test_images.size());
	}
}



void read_test_labels() {
	ifstream file;
	file.open("t10k-labels.idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			test_labels.push_back((double)label);
		};
		printf("%d, test labels success\n", test_labels.size());
	};
};



void ppprocess() {
	read_train_images();
	read_train_labels();
	read_test_images();
	read_test_labels();
};

//下面我将着手于对解码出的MNIST数据的处理

/*/查看获取的图像表示
void mmtt() {
	for (int j = 0; j < train_images[1].size(); j++) {
		cout << train_images[1][j] << " " << endl;
	}

	cout << train_labels[1] << endl;
};*/

vector<vector<double>> train_labels_onehot;
vector<vector<double>> test_labels_onehot;

//将标签用one-hot向量编码
void encode_onehot(const vector<double> &labels, vector<vector<double>>& accept_labels) {
	
	 // 用于存储 one-hot 向量
	// 读取标签并转化为 one-hot 向量
	for (auto x : labels) {
		vector<double> a_label_onehot(10, 0);
		int classIndex = static_cast<int>(x);
		a_label_onehot[classIndex] = 1;
		accept_labels.push_back(a_label_onehot);
	}
}


//获取one-hot表示
void obtain_onehot_labels(){

	encode_onehot(train_labels,train_labels_onehot);
    encode_onehot(test_labels,test_labels_onehot);

	cout <<"OneHot Vector Load Complete" << endl;
	//cout << train_labels_onehot[1][1] << "  " << test_labels_onehot[1][1] << endl;
}