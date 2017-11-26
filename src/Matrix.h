/*
 * Matrix.h
 *
 *  Created on: Nov 16, 2017
 *      Author: kole
 */
#ifndef MATRIX_H
#define MATRIX_H

#include "Utility.h"
#include <vector>
using json = nlohmann::json;

class Matrix {
public:
	Matrix(int n, int d);
	Matrix(int n, int d, float mu, float std); //matrix of random numbers from gaussian
	Matrix(json j); //matrix from JSON
	Matrix(const Matrix &m);//copy constructor
	Matrix& operator=(const Matrix &m);
	virtual ~Matrix();

	float getWeight(int row, int col);
	float getWeight(int ix);
	float getDWeight(int row, int col);
	float getDWeight(int ix);
	void setWeight(int row, int col, float v);
	void setWeight(int row, float v);
	void setDWeight(int row, int col, float v);
	void setDWeight(int row, float v);
	void setWeightsFromArray(std::vector<float>* array);
	void setDWeightsFromArray(std::vector<float>* array);
	void setColumn(Matrix *m, int col);
	Matrix* copy();
	void update(float alpha);
	void fillRandn(float mu, float std);
	void fillRand(float lo, float hi);
	void gradFillConst(float c);
	int length();
	int maxi();
	int samplei();
	int getRows();
	int getCols();
	json* toJSON();




private:
	int n; //rows
	int d; //columns
	std::vector<float>* w; //weights
	std::vector<float>* dw;
};

#endif
