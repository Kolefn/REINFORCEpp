/*
 * Graph.h
 *
 *  Created on: Nov 18, 2017
 *      Author: kole
 */
#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include "Matrix.h"
class Graph {
//@TODO should out in args be a pointer????
//generic struct of arguments a backprop function might need
struct args {
	Matrix* m1;
	Matrix* m2;
	Matrix* out;
	int ix;
	~args(){
		delete out;
	}
};

typedef void (Graph::*backprop_function)(args*);

public:
	Graph(bool needs_backprop=true);
	virtual ~Graph();
	void backward();
	Matrix* rowPluck(Matrix* m, int ix);
	Matrix* tanh(Matrix* m);
	Matrix* sigmoid(Matrix* m);
	Matrix* relu(Matrix* m);
	Matrix* mul(Matrix* m1, Matrix* m2);
	Matrix* add(Matrix* m1,Matrix* m2);
	Matrix* dot(Matrix* m1, Matrix* m2);
	Matrix* eltmul(Matrix* m1,Matrix* m2);
	Matrix* softmax(Matrix* m);





private:

	//@TODO should args be deleted after each backprop?
	void backRowPluck(args* a);
	void backTanh(args* a);
	void backSigmoid(args* a);
	void backRelu(args* a);
	void backMul(args* a);
	void backAdd(args* a);
	void backDot(args* a);
	void backEltmul(args* a);
	//softmax does not need a backward pass
	//'We will use the computed probabilities outside
	//to set the gradients directly on m'


	bool needs_backprop;

	//stores the functions which needs to be called upon backpropagation
	//the functions are stored as they are called in forward pass order
	std::vector<backprop_function> backprop;

	//stores the arguments for the backprop functions in the form of an
	//args struct. Not every function will use all the variables in the struct.
	std::vector<args*>backprop_args;

	std::vector<Matrix*>nobackprop_outs;
};

#endif
