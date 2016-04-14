#include "linear-model.h"

#include <cassert>
#include <tuple>

namespace ml {
namespace {

Vector linear_hypothesis(const Ref<const Matrix> &X,
                         const Ref<const Vector> &theta, double bias) {
  return X * theta + Vector::Ones(X.rows()) * bias;
}

Vector logistic_hypothesis(const Ref<const Matrix> &X,
                           const Ref<const Vector> &theta, double bias) {
  return sigmoid(linear_hypothesis(X, theta, bias));
}

template <typename H>
std::tuple<Vector, double>
minimize_mean_squared_error(const Ref<const Matrix> &X,
                            const Ref<const Vector> &y, const double alpha,
                            const H &hypothesis) {
  assert(X.rows() == y.rows());

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
  } while (!about_equal(previous_error, current_error)); // convergence
  return std::make_tuple(theta, bias);
}
}

LinearRegression::LinearRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LinearRegression::fit(const Ref<const Matrix> &X,
                           const Ref<const Vector> &y) {
  assert(X.rows() == y.rows());
  _is_fitted = true;

  auto result = minimize_mean_squared_error(X, y, _alpha, linear_hypothesis);
  _theta = std::get<0>(result);
  _bias = std::get<1>(result);
}

Vector LinearRegression::predict(const Ref<const Matrix> &X) const {
  assert(_is_fitted);
  return linear_hypothesis(X, _theta, _bias);
}

const Ref<const Vector> LinearRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}

LogisticRegression::LogisticRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LogisticRegression::fit(const Ref<const Matrix> &X,
                             const Ref<const IVector> &y) {
  assert(X.rows() == y.rows());
  _is_fitted = true;

  auto result = minimize_mean_squared_error(X, y.cast<double>(), _alpha,
                                            logistic_hypothesis);
  _theta = std::get<0>(result);
  _bias = std::get<1>(result);
}

Vector LogisticRegression::predict(const Ref<const Matrix> &X) const {
  assert(_is_fitted);
  return logistic_hypothesis(X, _theta, _bias);
}

const Ref<const Vector> LogisticRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}
}
