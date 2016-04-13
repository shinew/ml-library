#ifndef LINEAR_MODEl_H_
#define LINEAR_MODEl_H_

#include "model.h"
#include "utils.h"

namespace ml {

class LinearRegression : public Model {
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

class LogisticRegression : public Model {
public:
  LogisticRegression(double alpha);

  void fit(const Ref<const Matrix> &X, const Ref<const Vector> &y) override;

  Vector predict(const Ref<const Matrix> &X) const override;

  const Ref<const Vector> coefficients() const;

private:
  bool _is_fitted;
  const double _alpha;
  Vector _theta;
  double _bias;
};
}

#endif
