#include "linear-model.h"

#include <cassert>

namespace ml {
namespace {

Vector hypothesis(const Ref<const Matrix> &X, const Ref<const Vector> &theta,
                  double bias) {
  return X * theta + Vector::Ones(X.rows()) * bias;
}
}

double linear_regression_error(const Ref<const Matrix> &X,
                               const Ref<const Vector> &y,
                               const Ref<const Vector> &theta, double bias) {
  return mean_squared_error(hypothesis(X, theta, bias), y);
}

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

LogisticRegression::LogisticRegression(double alpha)
    : LinearRegression(alpha) {}

Vector LogisticRegression::predict(const Ref<const Matrix> &X) const {
  Vector y_predict = LinearRegression::predict(X);
  for (Eigen::DenseIndex i = 0; i < y_predict.size(); ++i) {
    y_predict(i) = sigmoid(y_predict(i));
  }
  return y_predict;
}

const Ref<const Vector> LogisticRegression::coefficients() const {
  return _theta;
}
}
