#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Core>
#include <cassert>

#include "regressor.h"
#include "utils.h"

namespace ml {

namespace {

Vector hypothesis(const Ref<const Matrix> &X, const Ref<const Vector> &theta,
                  double bias) {
  return X * theta + Vector::Ones(X.rows()) * bias;
}

double linear_regression_error(const Ref<const Matrix> &X,
                               const Ref<const Vector> &y,
                               const Ref<const Vector> &theta, double bias) {
  return mean_squared_error(hypothesis(X, theta, bias), y);
}

class LinearRegression : Regressor {
public:
  LinearRegression(double alpha);

  void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y) override;

  Vector predict(const Ref<const Matrix> &X) const override;

  const Ref<const Vector> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Vector _theta;
  double _bias;
};

LinearRegression::LinearRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LinearRegression::fit(const Ref<const Matrix> &X,
                           const Ref<const Vector> &y) {
  _is_fitted = true;

  _theta = Vector::Random(X.cols());
  double previous_error;
  double current_error = linear_regression_error(X, y, _theta, _bias);
  const auto X_transpose = X.transpose();
  do {
    previous_error = current_error;
    const auto h = hypothesis(X, _theta, _bias);
    const auto diff = X_transpose * (h - y);
    _theta -= _alpha * diff;
    current_error = linear_regression_error(X, y, _theta, _bias);
  } while (!about_equal(previous_error, current_error));
}

Vector LinearRegression::predict(const Ref<const Matrix> &X) const {
  assert(_is_fitted);
  return X * _theta;
}

const Ref<const Vector> LinearRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}
}
}

#endif
