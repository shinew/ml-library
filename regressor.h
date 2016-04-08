#ifndef REGRESSOR_H_
#define REGRESSOR_H_

#include <Eigen/Core>

namespace ml {

class Regressor {
public:
  virtual void fit(const Eigen::Ref<const Eigen::MatrixXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) = 0;
  virtual Eigen::VectorXd
  predict(const Eigen::Ref<const Eigen::MatrixXd> &x) const = 0;
};
}

#endif
