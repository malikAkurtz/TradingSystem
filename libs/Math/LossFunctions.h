#ifndef LOSS_FUNCS
#define LOSS_FUNCS

#include <vector>
#include "LinearAlgebra.h"
#include "LossFunctions.h"

#include <vector>
#include "LinearAlgebra.h"

double vectorizedModifiedSquarredError(const std::vector<double> &predictions, const std::vector<double> &labels);
double vectorizedLogLoss(const std::vector<double> &predictions, const std::vector<double> &labels);

#endif