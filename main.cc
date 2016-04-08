#include <Eigen/Core>
#include <iostream>
#include <vector>

#include "linear-regression.h"
#include "utils.h"

int main() {
  Eigen::VectorXd a(3);
  a << 1, 1, 2;
  std::cout << ml::mean_squared_error(a, a) << '\n';
  return 0;
}
