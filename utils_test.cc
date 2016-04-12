#include "utils.h"

#include "gtest/gtest.h"

namespace ml {
namespace {

TEST(MeanSquaredError, Basic) {
  Eigen::Vector3d a;
  Eigen::Vector3d b;
  b << 1.0, 2.0, 3.0;

  EXPECT_DOUBLE_EQ(14.0, mean_squared_error(a, b));
}

TEST(AboutEqual, Basic) {
  EXPECT_TRUE(about_equal(3.0, 3.0 + 1e-30));
  EXPECT_FALSE(about_equal(3.0, 3.0 + 1e-6));
}

TEST(Sigmoid, Basic) {
  // verified via WolframAlpha
  EXPECT_DOUBLE_EQ(0.880797077977882444059729141, sigmoid(2.0));
}
}
}
