#ifndef UTILS_H_
#define UTILS_H_

#include <Eigen/Core>
#include <cmath>

namespace ml {

double mean_squared_error(const Eigen::Ref<const Eigen::VectorXd> &a,
                          const Eigen::Ref<const Eigen::VectorXd> &b) {
  return (a - b).squaredNorm();
}

bool about_equal(double a, double b, double tol = 1e-50) {
  return fabs(a - b) < tol;
}
}

#endif
