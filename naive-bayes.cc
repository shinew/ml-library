#include "naive-bayes.h"

#include <algorithm>
#include <iostream>

namespace ml {

namespace {

// Assumes that y is a vector with elements 0 .. n.
int calculate_num_classes(const Ref<const IVector> &y) {
  int y_max = 0;
  for (Index i = 0; i < y.rows(); ++i) {
    y_max = std::max(y_max, y(i));
  }
  return y_max + 1;
}
}

MultinomialNaiveBayes::MultinomialNaiveBayes(double alpha)
    : _is_fitted(false), _alpha(alpha) {}

void MultinomialNaiveBayes::fit(const Ref<const IMatrix> &X,
                                const Ref<const IVector> &y) {
  assert(X.rows() == y.rows());
  _is_fitted = true;

  auto m = X.rows();
  const int num_classes = calculate_num_classes(y);
  assert(num_classes > 1);

  _theta_y = Vector::Zero(num_classes);
  for (Index i = 0; i < m; ++i) {
    _theta_y(y(i)) += 1;
  }
  // _theta_y(i) = # samples of class i.

  _theta_xy = Matrix::Zero(num_classes, X.cols());
  for (Index i = 0; i < m; ++i) {
    const int which_class = y(i);
    for (Index feature = 0; feature < X.cols(); ++feature) {
      if (X(i, feature) > 0) {
        _theta_xy(which_class, feature) += 1;
      }
    }
  }
  // _theta_xy(i)(j) = # samples with class i and x_j > 1.

  // normalize each (class, feature) -> probability
  for (Index which_class = 0; which_class < num_classes; ++which_class) {
    for (Index feature = 0; feature < X.cols(); ++feature) {
      _theta_xy(which_class, feature) =
          (_theta_xy(which_class, feature) + _alpha) /
          (_theta_y(which_class) + _alpha * num_classes);
    }
  }

  // normalize each class -> probability
  //_theta_y = _theta_y + Vector::Ones(num_classes) * _alpha) /
  //(m + _alpha * num_classes);
  _theta_y /= m;
}

Matrix MultinomialNaiveBayes::predict(const Ref<const IMatrix> &X) const {
  assert(_is_fitted);

  const int num_classes = _theta_y.size();
  Matrix output(X.rows(), num_classes);

  for (Index row = 0; row < X.rows(); ++row) {
    double denominator = 0.0;

    for (Index which_class = 0; which_class < num_classes; ++which_class) {
      double product = 1.0;
      for (Index feature = 0; feature < _theta_xy.cols(); ++feature) {
        if (X(row, feature) > 0) {
          product *= _theta_xy(which_class, feature);
        } else {
          product *= (1.0 - _theta_xy(which_class, feature));
        }
      }
      denominator += product * _theta_y(which_class);
    }

    for (Index which_class = 0; which_class < num_classes; ++which_class) {
      double p_x_given_y = 1.0;
      for (Index feature = 0; feature < _theta_xy.cols(); ++feature) {
        if (X(row, feature) > 0) {
          p_x_given_y *= _theta_xy(which_class, feature);
        } else {
          p_x_given_y *= (1.0 - _theta_xy(which_class, feature));
        }
      }
      output(row, which_class) =
          p_x_given_y * _theta_y(which_class) / denominator;
    }
  }

  return output;
}
}
