#ifndef LINEAR_MODEl_H_
#define LINEAR_MODEl_H_

#include "model.h"
#include "utils.h"

namespace ml {

double linear_regression_error(const Ref<const Matrix> &X,
                               const Ref<const Vector> &y,
                               const Ref<const Vector> &theta, double bias);

class LinearRegression : public Model {
public:
  LinearRegression(double alpha);

  void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y) override;

  Vector predict(const Ref<const Matrix> &X) const override;

  const Ref<const Vector> coefficients() const;

protected:
  Vector _theta;

private:
  bool _is_fitted;
  const double _alpha;
  double _bias;
};

class LogisticRegression : public LinearRegression {
public:
  LogisticRegression(double alpha);

  Vector predict(const Ref<const Matrix> &X) const override;

  const Ref<const Vector> coefficients() const;
};
}

#endif
