#include "sanity.h"

#include "gtest/gtest.h"

TEST(SanityTest, Sanity) {
  EXPECT_EQ(0, 0);
  EXPECT_EQ(42, f());
}
