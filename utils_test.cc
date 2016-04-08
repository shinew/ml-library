#include "utils.h"

#include "gtest/gtest.h"

namespace ml {
namespace {

TEST(MeanSquaredError, Arrays) {
  int a[] = {1, 2, 3};
  int b[] = {0, 0, 0};

  EXPECT_EQ(1 + 4 + 9, mean_squared_error(a, a + 3, b));
}

TEST(MeanSquaredError, STLVectors) {
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {0, 0, 0};

  EXPECT_EQ(1 + 4 + 9, mean_squared_error(a.cbegin(), a.cend(), b.cbegin()));
}
}
}
