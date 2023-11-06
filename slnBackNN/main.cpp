#pragma once

#include "layer.h"
#include "process.h"


using namespace std;


int main() {

	Layer myLayer;

	ppprocess();
	//mmtt();
	obtain_onehot_labels();

	
	myLayer.train(train_images);
	myLayer.predict(test_images, test_labels_onehot);

	cout << "\nHello World !\n" << endl;

	system("pause");

	return 0;
}
