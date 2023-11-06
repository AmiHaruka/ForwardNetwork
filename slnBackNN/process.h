#pragma once

#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <Windows.h>

using namespace std;

extern vector< vector<double> > train_images;
extern vector< vector<double> > test_images;
extern vector<vector<double>> train_labels_onehot;
extern vector<vector<double>> test_labels_onehot;

int reverse_int(int);

void read_train_images() ;

void read_train_labels();

void read_test_images();

void read_test_labels();

void ppprocess();

//void mmtt();

void encode_onehot(const vector<double> & ,vector<vector<double>> &);

void obtain_onehot_labels();
