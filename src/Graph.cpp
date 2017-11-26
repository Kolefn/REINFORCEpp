/*
 * Graph.cpp
 *
 *  Created on: Nov 18, 2017
 *      Author: kole
 */

#include "Graph.h"

Graph::Graph(bool needs_backprop): needs_backprop(needs_backprop){}

Graph::~Graph() {
	backprop.clear();

	//carefully delete outputs and args structures
	//don't delete other matrices (m1, m2) because they are inputs
	//to the graph. Leave it to outside classes to delete their matrices.
	if(needs_backprop){

		for(int i=backprop_args.size(); i >=0; i--){

			 //delete backprop_args[i];
		}
		backprop_args.clear();
	}else{
		for(int i=nobackprop_outs.size(); i >=0; i--){
			delete nobackprop_outs[i];
		}
		nobackprop_outs.clear();
	}


}


void Graph::backward(){
	for(int i = (int)backprop.size()-1; i >=0; i--){
		(this->*backprop[i])(backprop_args[i]);//call function with args
	}
}
//pluck a row of m with index ix and return it as a column vector
Matrix* Graph::rowPluck(Matrix* m, int ix){
	if(ix < 0 || ix > m->getRows()){
		std::cout << "Graph::rowPluck ix out of bounds" << std::endl;
		throw 0;
	}
	int d = m->getCols();
	Matrix* out = new Matrix(d,1);
	//copy row of m into out
	for(int i = 0; i < d; i++){
		out->setWeight(i, m->getWeight(d*ix+i));
	}

	if(needs_backprop){
		args* a = new args;
		a->ix = ix;
		a->m1 = m;
		a->out = out;
		backprop.push_back(&Graph::backRowPluck);
		backprop_args.push_back(a);
	}

	return out;
}


//update gradients based on forward pass out
void Graph::backRowPluck(args* a){
	int ix = a->ix;
	Matrix* m = a->m1;
	Matrix* out = a->out;
	int d = m->getCols();
	for(int i = 0; i < d; i++){
		int wi = d*ix+i;
		float v = m->getDWeight(wi) + out->getDWeight(i);
		m->setDWeight(wi,v);
	}

	delete a;
}

//tanh nonlinearity
Matrix* Graph::tanh(Matrix* m){
	int n = m->getRows();
	int d = m->getCols();
	Matrix* out = new Matrix(n,d);
	for(int i=0;i<n*d;i++){
		out->setWeight(i,std::tanh(m->getWeight(i)));
	}

	if(needs_backprop){
		args* a = new args;
		a->m1 = m;
		a->out = out;
		backprop.push_back(&Graph::backTanh);
		backprop_args.push_back(a);
	}else{
		nobackprop_outs.push_back(out);
	}

	return out;
}

void Graph::backTanh(args* a){
	Matrix* m = a->m1;
	Matrix* out = a->out;
	int n = m->length();
	for(int i =0; i < n; i++){
		float ow = out->getWeight(i);
		float mdw = m->getDWeight(i);
		mdw+= (1.0 - ow*ow) * out->getDWeight(i);
		m->setDWeight(i,mdw);
	}


	delete a;
}

Matrix* Graph::sigmoid(Matrix* m){
	int n = m->getRows();
	int d = m->getCols();
	Matrix* out = new Matrix(n,d);
	for(int i =0; i < n*d ; i++){
		out->setWeight(i, Utility::sig(m->getWeight(i)));
	}

	if(needs_backprop){
		args* a = new args;
		a->m1 = m;
		a->out = out;
		backprop.push_back(&Graph::backSigmoid);
		backprop_args.push_back(a);
	}else{
		nobackprop_outs.push_back(out);
	}

	return out;
}

void Graph::backSigmoid(args* a){
	Matrix* m = a->m1;
	Matrix* out = a->out;
	int n = m->length();
	for(int i=0; i < n; i++){
		float ow = out->getWeight(i);
		float mdw = m->getDWeight(i);
		mdw += ow * (1.0-ow) * out->getDWeight(i);
		m->setDWeight(i, mdw);
	}

	delete a;
}

//sets negative weights to 0
Matrix* Graph::relu(Matrix* m){
	int n = m->getRows();
	int d = m->getCols();
	Matrix* out = new Matrix(n,d);
	for(int i =0; i < n*d; i++){
		float ow = std::max(0.0f, m->getWeight(i)); //relu
		out->setWeight(i,ow);
	}

	if(needs_backprop){
		args* a = new args;
		a->m1 = m;
		a->out = out;
		backprop.push_back(&Graph::backRelu);
		backprop_args.push_back(a);
	}else{
		nobackprop_outs.push_back(out);
	}

	return out;
}

//add forward output gradients to matrix
//gradients if matrix weights above 0.
void Graph::backRelu(args* a){
	Matrix* m = a->m1;
	Matrix* out = a->out;
	int n = m->length();
	for(int i =0; i < n;i++){
		float mdw = m->getDWeight(i);
		mdw += m->getWeight(i) > 0 ? out->getDWeight(i) : 0.0;
		m->setDWeight(i,mdw);
	}

	delete a;
}


Matrix* Graph::mul(Matrix* m1, Matrix* m2){
	int z = m1->getCols();
	if(z != m2->getRows()){
		std::cout << "Graph::mul dimensions misaligned  " << z << " " << m2->getRows() << std::endl;
		throw 0;
	}

	int n = m1->getRows();
	int d = m2->getCols();
	Matrix* out = new Matrix(n,d);
	for(int i=0;i<n;i++){ //loop over rows of m1
		for(int j=0;j<d;j++){ //loop over cols of m2
			float dot = 0.0;
			for(int k=0;k<z;k++){ //dot product loop
				dot += m1->getWeight(z*i+k) * m2->getWeight(d*k+j);
			}
			out->setWeight(d*i+j, dot);
		}
	}

	if(needs_backprop){
			args* a = new args;
			a->m1 = m1;
			a->m2 = m2;
			a->out = out;
			backprop.push_back(&Graph::backMul);
			backprop_args.push_back(a);
	}else{
		nobackprop_outs.push_back(out);
	}

	return out;

}

void Graph::backMul(args* a){
	Matrix* m1 = a->m1;
	Matrix* m2 = a->m2;
	Matrix* out = a->out;
	int n = m1->getRows();
	int d = m2->getCols();
	int z = m1->getCols();

	for(int i=0; i < n; i++){ //loop over rows of m1
		for(int j=0; j < d; j++){ //loop over cols of m2
			for(int k=0; k < z; k++){ //dot product loop
				float b = out->getDWeight(d*i+j);

				float mdw = m1->getDWeight(z*i+k);
				float m2dw = m2->getDWeight(d*k+j);
				float mw = m1->getWeight(z*i+k);
				float m2w = m2->getWeight(d*k+j);
				m1->setDWeight(z*i+k, mdw+(m2w*b));
				m2->setDWeight(d*k+j, m2dw+(mw*b));

			}
		}

	}

	delete a;
}


Matrix* Graph::add(Matrix* m1, Matrix* m2){
	if(m1->length() != m2->length()){
		std::cout << "Graph::add matrices are not of equal length" << std::endl;
		throw 0;
	}

	int n = m1->getRows();
	int d = m1->getCols();

	Matrix* out = new Matrix(n,d);

	for(int i=0; i < n*d;i++){
		out->setWeight(i, m1->getWeight(i) + m2->getWeight(i));
	}

	if(needs_backprop){
		args* a = new args;
		a->m1 = m1;
		a->m2 = m2;
		a->out = out;
		backprop.push_back(&Graph::backAdd);
		backprop_args.push_back(a);
	}else{
		nobackprop_outs.push_back(out);
	}


	return out;

}

void Graph::backAdd(args* a){
		Matrix* m1 = a->m1;
		Matrix* m2 = a->m2;
		Matrix* out = a->out;
		int n = m1->length();
		for(int i=0; i < n; i++){
			float odw = out->getDWeight(i);
			float m1dw = m1->getDWeight(i);
			float m2dw = m2->getDWeight(i);
			m1->setDWeight(i,m1dw + odw);
			m2->setDWeight(i,m2dw + odw);
		}


		delete a;
}

Matrix* Graph::dot(Matrix* m1, Matrix* m2){
		if(m1->length() != m2->length()){
			std::cout << "Graph::dot matrices are not of equal length" << std::endl;
			throw 0;
		}
		int n = m1->length();
		Matrix* out = new Matrix(1,1);
		float dot = 0.0;
		for(int i =0;i<n;i++){
			dot+= m1->getWeight(i) * m2->getWeight(i);
		}

		out->setWeight(0,dot);

		if(needs_backprop){
				args* a = new args;
				a->m1 = m1;
				a->m2 = m2;
				a->out = out;
				backprop.push_back(&Graph::backDot);
				backprop_args.push_back(a);
		}else{
			nobackprop_outs.push_back(out);
		}

		return out;
}

void Graph::backDot(args* a){
	Matrix* m1 = a->m1;
	Matrix* m2 = a->m2;
	Matrix* out = a->out;
	int n = m1->length();
	for(int i=0; i<n;i++){
		float odw = out->getDWeight(0);
		float m1dw = m1->getDWeight(i);
		float m2dw = m2->getDWeight(i);
		m1dw += m2->getWeight(i) * odw;
		m2dw += m1->getWeight(i) * odw;
		m1->setDWeight(i, m1dw);
		m2->setDWeight(i, m2dw);
	}

	delete a;
}

Matrix* Graph::eltmul(Matrix* m1, Matrix* m2){
	if(m1->length() != m2->length()){
			std::cout << "Graph::eltmul matrices are not of equal length" << std::endl;
			throw 0;
		}
		int n = m1->getRows();
		int d = m1->getCols();

		Matrix* out = new Matrix(n,d);
		for(int i=0;i<n*d; i++){
			out->setWeight(i, m1->getWeight(i) * m2->getWeight(i));
		}

		if(needs_backprop){
			args* a = new args;
			a->m1 = m1;
			a->m2 = m2;
			a->out = out;
			backprop.push_back(&Graph::backEltmul);
			backprop_args.push_back(a);
		}else{
			nobackprop_outs.push_back(out);
		}


		return out;

}

void Graph::backEltmul(args * a){
	Matrix* m1 = a->m1;
	Matrix* m2 = a->m2;
	Matrix* out = a->out;
	int n = m1->length();
	for(int i=0; i<n;i++){
		float odw = out->getDWeight(i);
		float m1dw = m1->getDWeight(i);
		float m2dw = m2->getDWeight(i);
		m1dw += m2->getWeight(i) * odw;
		m2dw += m1->getWeight(i) * odw;
		m1->setDWeight(i, m1dw);
		m2->setDWeight(i, m2dw);
	}

	delete a;
}


Matrix* Graph::softmax(Matrix* m){
	int n = m->getRows();
	int d = m->getCols();
	Matrix* out = new Matrix(n,d);
	float maxval = -999999;
	for(int i = 0; i<n*d;i++){
		float mw = m->getWeight(i);
		if(mw > maxval){
			maxval = mw;
		}
	}
	float s = 0.0;
	for(int i = 0; i<n*d;i++){
		float ow = std::exp(m->getWeight(i) - maxval);
		out->setWeight(i,ow);
		s += ow;
	}
	for(int i = 0; i < n*d; i++){
		float ow = out->getWeight(i);
		out->setWeight(i, ow/s);
	}

	//no backward pass needed here
	//since 'We will use the computed probabilities outside
	//to set gradients directly on m'

	nobackprop_outs.push_back(out);

	return out;
}
