#include "linear-model.h"

#include <cassert>
#include <tuple>

namespace ml {
namespace {

Vector hypothesis(const Ref<const Matrix> &X, const Ref<const Vector> &theta,
                  double bias) {
  return X * theta + Vector::Ones(X.rows()) * bias;
}

std::tuple<Vector, double>
minimize_mean_squared_error(const Ref<const Matrix> &X,
                            const Ref<const Vector> &y, const double alpha) {
  Vector theta = Vector::Random(X.cols());
  double bias = 0.0;
  double previous_error;
  double current_error = mean_squared_error(hypothesis(X, theta, bias), y);
  const auto X_transpose = X.transpose();
  do {
    previous_error = current_error;
    const auto h = hypothesis(X, theta, bias);
    const auto diff = X_transpose * (h - y);
    theta -= alpha * diff;
    current_error = mean_squared_error(hypothesis(X, theta, bias), y);
  } while (!about_equal(previous_error, current_error));
  return std::make_tuple(theta, bias);
}
}

LinearRegression::LinearRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LinearRegression::fit(const Ref<const Matrix> &X,
                           const Ref<const Vector> &y) {
  _is_fitted = true;

  auto result = minimize_mean_squared_error(X, y, _alpha);
  _theta = std::get<0>(result);
  _bias = std::get<1>(result);
}

Vector LinearRegression::predict(const Ref<const Matrix> &X) const {
  assert(_is_fitted);
  return hypothesis(X, _theta, _bias);
}

const Ref<const Vector> LinearRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}

LogisticRegression::LogisticRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LogisticRegression::fit(const Ref<const Matrix> &X,
                             const Ref<const Vector> &y) {
  _is_fitted = true;

  auto result = minimize_mean_squared_error(X, y, _alpha);
  _theta = std::get<0>(result);
  _bias = std::get<1>(result);
}

Vector LogisticRegression::predict(const Ref<const Matrix> &X) const {
  assert(_is_fitted);
  return sigmoid(hypothesis(X, _theta, _bias));
}

const Ref<const Vector> LogisticRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}
}
