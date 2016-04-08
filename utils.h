#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Core>
#include <cmath>

namespace ml {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Eigen::Ref;

double mean_squared_error(const Ref<const Vector> &a,
                          const Ref<const Vector> &b) {
  return (a - b).squaredNorm();
}

bool about_equal(double a, double b, double tol = 1e-50) {
  return fabs(a - b) < tol;
}
}

#endif
