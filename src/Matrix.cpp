/*
 * Matrix.cpp
 *
 *  Created on: Nov 16, 2017
 *      Author: kole
 */

#include "Matrix.h"

//n is rows, d is columns
Matrix::Matrix(int n, int d): n(n), d(d) {

	w = Utility::zeros(n*d);
	dw = Utility::zeros(n*d);

}

//random matrix
Matrix::Matrix(int n, int d, float mu, float std): n(n), d(d){
	w = Utility::zeros(n*d);
	dw = Utility::zeros(n*d);
	this->fillRandn(mu, std);
}

//matrix build from JSON
Matrix::Matrix(json j){
	n = j["n"];
	d = j["d"];
	w = Utility::zeros(n*d);
	dw = Utility::zeros(n*d);
	std::vector<float> jw = j["w"];
	std::vector<float> jdw = j["dw"];


	//copy weights
	for(int i=0, z=n*d; i < z; i++){
		(*w)[i] = jw[i];
		(*dw)[i] = jdw[i];
	}


}
//careful accessor function
float Matrix::getWeight(int row, int col){
	int ix = (d * row) + col;
	if(ix >= 0 && ix < w->size()){
		return w->at(ix);
	}else {
		std::cout << "Matrix::setWeight row/col not in matrix" << std::endl;
		throw 0;
	}
}

float Matrix::getWeight(int ix){
	if(ix >= 0 && ix < w->size()){
		return w->at(ix);
	}else{
		std::cout << "Matrix::getWeight ix not in matrix" << std::endl;
		throw 0;
	}
}

//careful accessor function
float Matrix::getDWeight(int row, int col){
	int ix = (d * row) + col;
	if(ix >= 0 && ix < dw->size()){
		return dw->at(ix);
	}else {
		std::cout << "Matrix::setDWeight row/col not in matrix" << std::endl;
		throw 0;
	}
}

float Matrix::getDWeight(int ix){
	if(ix >= 0 && ix < dw->size()){
		return dw->at(ix);
	}else{
		std::cout << "Matrix::getDWeight row/col not in matrix" << std::endl;
		throw 0;
	}
}

void Matrix::setWeight(int row, int col, float v){
	//careful set function
	int ix = (d * row) + col;
	if(ix >= 0 && ix < w->size()){
		(*w)[ix] = v;
	}else{
		std::cout << "Matrix::setWeight row/col not in matrix" << std::endl;
		throw 0;
	}

}

void Matrix::setWeight(int ix,float v){
	//careful set function
	if(ix >= 0 && ix < w->size()){
		(*w)[ix] = v;
	}else{
		std::cout << "Matrix::setWeight ix not in matrix" << std::endl;
		throw 0;
	}

}

void Matrix::setDWeight(int row, int col, float v){
	//careful set function
	int ix = (d * row) + col;
	if(ix >= 0 && ix < dw->size()){
		(*dw)[ix] = v;
	}else{
		std::cout << "Matrix::setDWeight row/col not in matrix"<< std::endl;
		throw 0;
	}

}

void Matrix::setDWeight(int ix,float v){
	//careful set function
	if(ix >= 0 && ix < dw->size()){
		(*dw)[ix] = v;
	}else{
		std::cout << "Matrix::setDWeight ix not in matrix"<< std::endl;
		throw 0;
	}

}

void Matrix::setWeightsFromArray(std::vector<float>* array){
	for(int i = 0, z = (int)array->size(); i < z; i++){
		(*w)[i] = array->at(i);
	}
}

void Matrix::setDWeightsFromArray(std::vector<float>* array){
	for(int i = 0, z = (int)array->size(); i < z; i++){
		(*dw)[i] = array->at(i);
	}
}

//sets an entire column of this matrix with the 0...n values of input matrix
//yes that is weird.
void Matrix::setColumn(Matrix *m, int col){
	for(int q=0, z=m->length(); q<z; q++){
		int colRowIdx = (d*q) + col;
		(*w)[colRowIdx] = m->getWeight(q);
	}
}

Matrix* Matrix::copy(){
	Matrix *a = new Matrix(n,d);
	a->setWeightsFromArray(w);
	a->setDWeightsFromArray(dw);
	return a;
}
//copy constructor
Matrix::Matrix(const Matrix &m){
	n = m.n;
	d = m.d;
	w = Utility::zeros(n*d);
	dw = Utility::zeros(n*d);
	setWeightsFromArray(m.w);
	setDWeightsFromArray(m.dw);
}

Matrix& Matrix::operator=(const Matrix &m){
	n = m.n;
	d = m.d;
	w = Utility::zeros(n*d);
	dw = Utility::zeros(n*d);
	setWeightsFromArray(m.w);
	setDWeightsFromArray(m.dw);
	return *this;
}

void Matrix::update(float alpha){
	for(int i = 0,z=(int)w->size(); i < z; i++){
		if(dw->at(i) != 0){
			(*w)[i] += -alpha * dw->at(i);
			(*dw)[i] = 0;
		}
	}
}

void Matrix::fillRandn(float mu, float std) {
	for(int i = 0, z = (int)w->size(); i < z; i++){
		(*w)[i] = Utility::randomN(mu, std);
	}
}

void Matrix::fillRand(float lo, float hi){
	for(int i = 0, z=(int)w->size(); i < z; i++){
		(*w)[i] = Utility::randomFloat(lo, hi);
	}
}
//initialize delta weights with constant
void Matrix::gradFillConst(float c){
	for(int i=0, z=(int)dw->size();i < z; i++){
		(*dw)[i] = c;
	}
}

json* Matrix::toJSON(){
	json* j = new json;
	(*j)["n"] = n;
	(*j)["d"] = d;
	(*j)["w"] = (*w);
	(*j)["dw"] = (*dw);

	return j;
}

int Matrix::length(){
	return (int)w->size();
}

//returns the index of max weight
int Matrix::maxi(){
	float maxv = (*w)[0];
	int maxix = 0;
	for(int i = 1, z = (int)w->size(); i < z; i++){
		float v = (*w)[i];
		if(v > maxv){
			maxix = i;
			maxv = v;
		}
	}

	return maxix;
}
//sample argmax from weights
//assuming weights are probabilities which sum = 1
//if not, returns last weight
int Matrix::samplei(){
	float r = Utility::randomFloat(0,1);
	float x = 0.0;
	for(int i = 1, z = (int)w->size(); i < z; i++){
		x += (*w)[i];
		if(x > r){return i;}
	}

	return (int)w->size() - 1;
}


int Matrix::getRows(){
	return n;
}

int Matrix::getCols(){
	return d;
}

Matrix::~Matrix() {
	delete w;
	delete dw;

}

