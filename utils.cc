#include "utils.h"

#include <cmath>
#include <utility>

namespace ml {

double mean_squared_error(const Ref<const Vector> &a,
                          const Ref<const Vector> &b) {
  return (a - b).squaredNorm();
}

bool about_equal(double a, double b, double tol) { return fabs(a - b) < tol; }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

Vector sigmoid(const Vector &x) {
  return x.unaryExpr([](double v) { return sigmoid(v); });
}

Vector softmax(const Vector &x) {
  Vector numerators = x.unaryExpr([](double v) { return std::exp(v); });
  double denominator = numerators.sum();
  numerators /= denominator;
  return numerators;
}
}
