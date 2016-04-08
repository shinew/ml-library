#ifndef MODEL_H_
#define MODEL_H_

#include <Eigen/Dense>

namespace ml {

using Eigen::Dense::

class Model {
public:
  virtual void fit();
  virtual void predict();
};
}

#endif
