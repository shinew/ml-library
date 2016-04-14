#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Core>

namespace ml {

using Vector = Eigen::VectorXd;
using IVector = Eigen::VectorXi;
using Matrix = Eigen::MatrixXd;
using IMatrix = Eigen::MatrixXi;
using Index = Eigen::DenseIndex;
using Eigen::Ref;

double mean_squared_error(const Ref<const Vector> &a,
                          const Ref<const Vector> &b);

bool about_equal(double a, double b, double tol = 1e-10);

double sigmoid(double x);

Vector sigmoid(const Vector &x);

Vector softmax(const Vector &x);
}

#endif
