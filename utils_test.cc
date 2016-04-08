#include "utils.h"

#include "gtest/gtest.h"
#include <Eigen/Core>

namespace ml {
namespace {

TEST(MeanSquaredError, Basic) {
  Eigen::Vector3d a;
  Eigen::Vector3d b;
  b << 1.0, 2.0, 3.0;

  EXPECT_DOUBLE_EQ(14.0, mean_squared_error(a, b));
}
}
}
