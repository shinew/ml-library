#include "utils.h"

#include <cmath>

namespace ml {

double mean_squared_error(const Ref<const Vector> &a,
                          const Ref<const Vector> &b) {
  return (a - b).squaredNorm();
}

bool about_equal(double a, double b, double tol) { return fabs(a - b) < tol; }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}
