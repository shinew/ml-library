#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Core>
#include <cassert>

#include "regressor.h"
#include "utils.h"

namespace ml {

namespace {

Eigen::VectorXd hypothesis(const Eigen::Ref<const Eigen::MatrixXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &theta,
                           double bias) {
  return x * theta + Eigen::VectorXd::Ones(theta.rows()) * bias;
}

double linear_regression_error(const Eigen::Ref<const Eigen::MatrixXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &y,
                               const Eigen::Ref<const Eigen::VectorXd> &theta,
                               double bias) {
  return mean_squared_error(hypothesis(x, theta, bias), y);
}

class LinearRegression : Regressor {
public:
  LinearRegression(double alpha);

  void fit(const Eigen::Ref<const Eigen::MatrixXd> &x,
           const Eigen::Ref<const Eigen::VectorXd> &y) override;

  Eigen::VectorXd
  predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const override;

  const Eigen::Ref<const Eigen::VectorXd> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Eigen::VectorXd _theta;
  double _bias;
};

LinearRegression::LinearRegression(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void LinearRegression::fit(const Eigen::Ref<const Eigen::MatrixXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &y) {
  _is_fitted = true;

  _theta = Eigen::VectorXd::Random(x.rows());
  double previous_error;
  double current_error = linear_regression_error(x, y, _theta, _bias);
  do {
    previous_error = current_error;
    auto h = hypothesis(x, _theta, _bias);
    auto diff = x * (h - y);
    _theta -= _alpha * diff;
    current_error = linear_regression_error(x, y, _theta, _bias);
  } while (!about_equal(previous_error, current_error));
}

Eigen::VectorXd
LinearRegression::predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const {
  assert(_is_fitted);
  return x * _theta;
}

const Eigen::Ref<const Eigen::VectorXd> LinearRegression::coefficients() const {
  assert(_is_fitted);
  return _theta;
}
}
}

#endif
