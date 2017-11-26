/*
 * DQNAgent.cpp
 *
 *  Created on: Nov 19, 2017
 *      Author: kole
 */

#include "DQNAgent.h"

DQNAgent::DQNAgent(){}

void DQNAgent::set(int _NUM_STATE_FEATURES, int _MAX_NUM_ACTIONS, int _NUM_HIDDEN_UNITS,
              float _gamma, float _epsilon, float _alpha, int _experience_add_every,
              int _EXPERIENCE_SIZE, int _learning_steps_per_iteration, float _tderror_clamp){
    ns = _NUM_STATE_FEATURES;
    na = _MAX_NUM_ACTIONS;
    nh = _NUM_HIDDEN_UNITS;
    gamma = _gamma;
    epsilon = _epsilon;
    alpha = _alpha;
    experience_add_every = _experience_add_every;
    EXPERIENCE_SIZE = _EXPERIENCE_SIZE;
    learning_steps_per_iteration = _learning_steps_per_iteration;
    tderror_clamp = _tderror_clamp;
    
    //the model
    Matrix* W1 = new Matrix(nh,ns,0,0.01); //input-to-hidden layer
    net.add("W1",W1);
    Matrix* b1 = new Matrix(nh,1,0,0.1); //hidden layer
    net.add("b1",b1);
    Matrix* W2 = new Matrix(na,nh,0,0.01); //hidden-to-output layer
    net.add("W2",W2);
    Matrix* b2 = new Matrix(na,1,0,0.01); //output layer
    net.add("b2",b2);
    
    //exp vector is empty at this point
    expi = 0;
    
    t = 0;
    
    tderror = 0;
    
    exp = std::vector<SARSA>();
    lastG = new Graph(false);
};

DQNAgent::DQNAgent(agent_options opt) {

	nh = opt.NUM_HIDDEN_UNITS;
	ns = opt.NUM_STATE_FEATURES;
	na = opt.MAX_NUM_ACTIONS;

	gamma = opt.gamma;
	epsilon = opt.epsilon;
	alpha = opt.alpha;
	experience_add_every = opt.experience_add_every;
	EXPERIENCE_SIZE = opt.EXPERIENCE_SIZE;
	learning_steps_per_iteration = opt.learning_steps_per_iteration;
	tderror_clamp = opt.tderror_clamp;

	//the model
	Matrix* W1 = new Matrix(nh,ns,0,0.01); //input-to-hidden layer
	net.add("W1",W1);
	Matrix* b1 = new Matrix(nh,1,0,0.1); //hidden layer
	net.add("b1",b1);
	Matrix* W2 = new Matrix(na,nh,0,0.01); //hidden-to-output layer
	net.add("W2",W2);
	Matrix* b2 = new Matrix(na,1,0,0.01); //output layer
	net.add("b2",b2);

	//exp vector is empty at this point
	expi = 0;

	t = 0;

	tderror = 0;

	exp = std::vector<SARSA>();
	lastG = new Graph(false);

}

DQNAgent::DQNAgent(json opt) {

	gamma = opt["gamma"];
	epsilon = opt["epsilon"];
	alpha = opt["alpha"];
	experience_add_every = opt["experience_add_every"];
	EXPERIENCE_SIZE = opt["EXPERIENCE_SIZE"];
	learning_steps_per_iteration = opt["learning_steps_per_iteration"];
	tderror_clamp = opt["tderror_clamp"];

	nh = opt["nh"];
	ns = opt["ns"];
	na = opt["na"];

	net.fromJSON(opt["net"]);

	std::cout << net.getMatrix("W1")->length() << "  " << net.getMatrix("b1")->length() << std::endl;
	//exp vector is empty at this point
	expi = 0;

	t = 0;

	tderror = 0;

	exp = std::vector<SARSA>();
	lastG = new Graph(false);
}

DQNAgent::~DQNAgent() {
	net.~Network(); //delete matrix pointers
    lastG->~Graph();
}


Matrix* DQNAgent::forwardQ(Matrix* s, bool needs_backprop){
	Graph* G = new Graph(needs_backprop);
	Matrix* a1matmul = G->mul(net.getMatrix("W1"),s);
	Matrix* a1mat = G->add(a1matmul,net.getMatrix("b1"));
	Matrix* h1mat = G->tanh(a1mat);
	Matrix* a2matmul = G->mul(net.getMatrix("W2"),h1mat);
	Matrix* a2mat = G->add(a2matmul, net.getMatrix("b2"));



	delete lastG; //don't need the graph it points to anymore
	//We don't delete the outputted matrices here because they're
	//used by the graph again in backprop. They will be deleted when forwardQ
	//is called again and the graph is replaced below.
	lastG = G;
	return a2mat;

}

int DQNAgent::act(std::vector<float> slist){
	//convert state to a matrix column vector
	Matrix* s = new Matrix(ns,1);
	s->setWeightsFromArray(&slist);

	int a = 0; //action

	//epsilon greedy policy
	//will agent take random action or calculated action?
	if(Utility::randomFloat(0,1) < epsilon){
		a = Utility::randomInt(0, na); //random action
	}else {
		//action based on Q function
		Matrix* amat = DQNAgent::forwardQ(s,false); //action matrix
		a = amat->maxi();  //index of largest weight
	}

	sarsa.s0 = sarsa.s1;	//prev state
	sarsa.a0 = sarsa.a1;	//prev action
	sarsa.s1 = *s;	//current state - dereferenced
	sarsa.a1 = a;	//current action
	//amat matrix will be deleted upon lastG deletion in next forwardQ call


	return a;

}

void DQNAgent::learn(float r1){
	//update the Q function with new reward
	//as long as there is a previous reward and the learning rate is above zero.
	//no learning rate, no learning!

	if(sarsa.r0 != std::numeric_limits<float>::infinity() && alpha > 0){

		//determine how unexpected the values of SARSA are to this agent
		float _tderror = DQNAgent::learnFromTuple(sarsa);
		tderror = _tderror; //"measure of surprise"


		//decide if this experience should be kept in the replay vector
		if(t % experience_add_every == 0){ //if amount of time has passed

			if(exp.size() == expi){
				exp.push_back(sarsa); //add
			}else{
				exp[expi] = sarsa; //replace
			}
			expi++;
			if(expi > EXPERIENCE_SIZE){expi = 0;} //roll over to start replacing old memory
		}
		t++; //time tick



		//sample additional experience from replay memory and learn from it
		for(int k=0; k < learning_steps_per_iteration;k++){
			int ri = Utility::randomInt(0,(int)exp.size()); //TODO priority sweeps
			SARSA e = exp[ri]; //random experience
			DQNAgent::learnFromTuple(e);
		}




	}

	sarsa.r0 = r1; //store for next update
}

//by default will learn from this->sarsa
float DQNAgent::learnFromTuple(SARSA _sarsa){

	//mathematically summary: Q(s,a) = r + gamma * max_a' * Q(s',a')
	//compute the target Q value with dereferenced s1
	Matrix* _s1 = _sarsa.s1.copy();
	Matrix* tmat = DQNAgent::forwardQ(_s1,false);
	float qmax = _sarsa.r0 + gamma * tmat->getWeight(tmat->maxi());

	//now predict
	//@TODO not sure if need to create copy here
	Matrix* _s0 = _sarsa.s0.copy(); //create this so it can be used in backprop
	Matrix* pred = DQNAgent::forwardQ(_s0, true);

	float _tderror = pred->getWeight(_sarsa.a0) - qmax;
	float clamp = tderror_clamp;
	if(std::abs(_tderror) > clamp){ //huber loss to robustify
		if(_tderror > clamp) _tderror = clamp;
		if(_tderror < -clamp) _tderror = -clamp;
	}

	pred->setDWeight(_sarsa.a0, _tderror);
	lastG->backward(); //computer gradients on net

	delete _s1;
	delete _s0;

	//update net
	net.update(alpha);
	return _tderror;
}





json* DQNAgent::toJSON(){
	json* j = new json();
	(*j)["nh"] = nh;
	(*j)["ns"] = ns;
	(*j)["na"] = na;
	(*j)["gamma"] = gamma;
	(*j)["epsilon"] = epsilon;
	(*j)["alpha"] = alpha;
	(*j)["experience_add_every"] = experience_add_every;
	(*j)["EXPERIENCE_SIZE"] = EXPERIENCE_SIZE;
	(*j)["learning_steps_per_iteration"] = learning_steps_per_iteration;
	(*j)["tderror_clamp"] = tderror_clamp;
	json* temp = net.toJSON();
	(*j)["net"] = (*temp);
	delete temp; //never used again so delete

	return j;
}


void DQNAgent::adjustAlpha(float x){
    alpha -= x;
    alpha = std::max(0.0f, alpha);
}

void DQNAgent::adjustEpsilon(float x){
    epsilon -= x;
    epsilon = std::max(0.0f, epsilon);
}
void DQNAgent::adjustGamma(float x){
    gamma -= x;
    gamma = std::min(0.9f, gamma);
}


int DQNAgent::getStateFeatures(){
    return ns;
}



