#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Core>
#include <cassert>

#include "regressor.h"
#include "utils.h"

namespace ml {

namespace {

Vector hypothesis(const Ref<const Matrix> &x, const Ref<const Vector> &theta,
                  double bias) {
  return x * theta + Vector::Ones(x.rows()) * bias;
}

double linear_regression_error(const Ref<const Matrix> &x,
                               const Ref<const Vector> &y,
                               const Ref<const Vector> &theta, double bias) {
  return mean_squared_error(hypothesis(x, theta, bias), y);
}

class LinearRegression : Regressor {
public:
  LinearRegression(double alpha);

  void fit(const Ref<const Matrix> &x, const Ref<const Vector> &y) override;

  Vector predict(const Ref<const Matrix> &x) const override;

  const Ref<const Vector> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Vector _theta;
  double _bias;
};

LinearRegression::LinearRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LinearRegression::fit(const Ref<const Matrix> &x,
                           const Ref<const Vector> &y) {
  _is_fitted = true;

  _theta = Vector::Random(x.cols());
  double previous_error;
  double current_error = linear_regression_error(x, y, _theta, _bias);
  auto x_transpose = x.transpose();
  do {
    previous_error = current_error;
    auto h = hypothesis(x, _theta, _bias);
    auto diff = x_transpose * (h - y);
    _theta -= _alpha * diff;
    current_error = linear_regression_error(x, y, _theta, _bias);
  } while (!about_equal(previous_error, current_error));
}

Vector LinearRegression::predict(const Ref<const Matrix> &x) const {
  assert(_is_fitted);
  return x * _theta;
}

const Ref<const Vector> LinearRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}
}
}

#endif
