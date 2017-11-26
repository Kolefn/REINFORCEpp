/*
 //  API.hpp
 //  RPP
 //
 //  Created by Kole Nunley on 11/22/17.
 //  Copyright © 2017 Kole Nunley. All rights reserved.
 

 The api exposes several functions so other programs can create agent with the RPP bundle.
 */
#pragma once
#ifndef API_hpp
#define API_hpp

#include "Utility.h"
#include "DQNAgent.h"

extern "C" {
    DQNAgent* addAgent(int _NUM_STATE_FEATURES, int _MAX_NUM_ACTIONS, int _NUM_HIDDEN_UNITS,
                  float _gamma, float _epsilon, float _alpha, int _experience_add_every,
                  int _EXPERIENCE_SIZE, int _learning_steps_per_iteration, float _tderror_clamp);
    void deleteAgent(DQNAgent* agent);
    
    void rewardAgent(DQNAgent* agent, float reward, int dAlpha=0, int dEpsilon=0);
    int  getActionFromAgent(DQNAgent* agent, float* state);
}




#endif /* API_hpp */
