#ifndef LINEAR_MODEl_H_
#define LINEAR_MODEl_H_

#include "utils.h"

namespace ml {

class LinearRegression {
public:
  LinearRegression(double alpha);

  void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y);

  Vector predict(const Ref<const Matrix> &X) const;

  const Ref<const Vector> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Vector _theta;
  double _bias;
};

class LogisticRegression {
public:
  LogisticRegression(double alpha);

  void fit(const Ref<const Matrix> &X, const Ref<const IVector> &y);

  Vector predict(const Ref<const Matrix> &X) const;

  const Ref<const Vector> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Vector _theta;
  double _bias;
};
}

#endif
