/*
 * nn.c
 *
 *  Created on: Dec 25, 2025
 *      Author: Jon
 */

#include "nn.h"

float relu(float x) {
	return (x > 0.0f) ? x : 0.0f;
}

float W1[3][5] = {
  { -7.507755280f, -0.623517454f,  7.667947292f, -0.022908390f, -10.418303490f },
  { -7.778380871f, -0.103291929f, -9.772282600f, -0.108166635f,  8.484452248f },
  {  6.848341942f, -0.136626780f,  9.725394249f, -0.108934104f,  7.597446442f },
};

float B1[5] = {
  5.946643829f, 0.000000000f, -0.639164507f, 0.000000000f, 1.008382678f
};

float W2[5][8] = {
  {  3.377823591f, -15.364077568f,  6.693141937f, -18.613655090f,  7.402841091f, -0.741920710f, -10.774998665f, -2.134330273f },
  {  0.125155985f,  -0.067114890f, -0.282764167f,  -0.365252912f,  0.093472660f,  0.465394020f,   0.232686639f,  0.471728802f },
  { -11.118428230f,  5.985622883f, -7.554813862f,   6.439907551f, -8.391246796f,  2.314197540f, -17.404479980f,  1.063245416f },
  { -0.636800230f,   0.539791584f, -0.367890149f,  -0.607674956f,  0.599651575f, -0.244209290f,   0.295462549f,  0.123288274f },
  { -3.730190516f,   1.266279101f, -15.921719551f, -6.638589859f, -6.037054062f,  2.664514065f,   4.671674252f,  1.120581150f },
};

float B2[8] = {
  11.663460732f, -3.252909660f, -1.463615060f, 1.448936224f,
 -10.632442474f, -15.515432358f, -6.048957348f, 14.183552742f
};

char* CLASS_NAMES[8] = {
 "Light Touch",
 "Hard Touch",
 "Left Light",
 "Left Hard",
 "Right Light",
 "Right Hard",
 "Middle",
 "Indeterminate"
};

static int classify(float output[8]) {
  int max_index = 0;
  float max = output[0];

  for (int i = 1; i < 8; i++) {
    if (output[i] > max) {
    	max = output[i];
    	max_index = i;
    }
  }
  return max_index;
}

char* nn_predict(float in[3]) {
  float h[5];
  float output[8];

  for (int i = 0; i < 5; i++) {

	float sum = 0;

	for (int j = 0; j < 3; j++) {
		sum += W1[j][i] * in[j];
	}

	sum += B1[i];

	h[i] = relu(sum);
  }

  for (int i = 0; i < 8; i++) {

	float sum = 0;

	for (int j = 0; j < 5; j++) {
		sum += W2[j][i] * h[j];
	}

	sum += B2[i];

	output[i] = sum;
  }

  int classification = classify(output);

  return CLASS_NAMES[classification];
}
