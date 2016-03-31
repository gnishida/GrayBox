/**
 * This is a sample implementation of the Gray Box model described in
 * "An Inverse Gray-Box Model for Transient Building Load Prediction"
 * by Braun and Chaturvedi.
 *
 * @author Gen Nishida
 * @date 3/7/2016
 */

#include <iostream>
#include "GrayBox.h"
#include <fstream>

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <csv file name>" << std::endl;
		return -1;
	}

	GrayBox gb;
	gb.loadTrainingData(argv[1]);

	cv::Mat_<double> predictedY;
	gb.forward(predictedY);

	std::ofstream out;
	out.open("result.txt");
	for (int c = 0; c < predictedY.cols; ++c) {
		out << predictedY(0, c) << "," << predictedY(1, c) << std::endl;
	}
	out.close();

	return 0;
}