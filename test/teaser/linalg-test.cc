/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <iostream>
#include <random>

#include <Eigen/Core>

#include "teaser/linalg.h"

TEST(LinalgTest, HatMap) {
  {
    // all zeros
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 0, 0, 0;
    Eigen::Matrix3f exp;
    exp.setZero();
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero cases
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 1, 2, 3;
    Eigen::Matrix3f exp;
    // clang-format off
    exp << 0, -3, 2,
           3,  0,-1,
          -2,  1, 0;
    // clang-format on
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // negative
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 0, 2, -3;
    Eigen::Matrix3f exp;
    // clang-format off
    exp << 0,   3, 2,
           -3,  0, 0,
           -2,  0, 0;
    // clang-format on
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
}

TEST(LinalgTest, VectorKronFixedSize) {
  {
    // a simple case
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 1, 1, 1;
    Eigen::Matrix<float, 3, 1> v2;
    v2 << 1, 1, 1;
    Eigen::Matrix<float, 9, 1> exp = Eigen::Matrix<float, 9, 1>::Ones(9, 1);

    // compute the actual result
    Eigen::Matrix<float, 9, 1> act;
    teaser::vectorKron<float, 3, 3>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero vectors
    Eigen::Matrix<float, 2, 1> v1;
    v1 << 1, 4;
    Eigen::Matrix<float, 2, 1> v2;
    v2 << 3, 2;
    Eigen::Matrix<float, 4, 1> exp;
    exp << 3, 2, 12, 8;

    // compute the actual result
    Eigen::Matrix<float, 4, 1> act;
    teaser::vectorKron<float, 2, 2>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 1, -3, 3;
    Eigen::Matrix<float, 4, 1> v2;
    v2 << 3, -1, 9, -10;
    Eigen::Matrix<float, 12, 1> exp;
    exp << 3, -1, 9, -10, -9, 3, -27, 30, 9, -3, 27, -30;

    // compute the actual result
    Eigen::Matrix<float, 12, 1> act;
    teaser::vectorKron<float, 3, 4>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths (all zero)
    Eigen::Matrix<float, 3, 1> v1;
    v1 << 0, 0, 0;
    Eigen::Matrix<float, 4, 1> v2;
    v2 << 3, -1, 9, -10;
    Eigen::Matrix<float, 12, 1> exp;
    exp.setZero();

    // compute the actual result
    Eigen::Matrix<float, 12, 1> act;
    teaser::vectorKron<float, 3, 4>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
}

TEST(LinalgTest, VectorKronDynamicSize) {
  {
    // a simple case
    Eigen::VectorXf v1(3, 1);
    v1 << 1, 1, 1;
    Eigen::VectorXf v2(3, 1);
    v2 << 1, 1, 1;
    Eigen::VectorXf exp(9, 1);
    exp.setOnes();

    // compute the actual result
    Eigen::VectorXf act = teaser::vectorKron<float, 3, 3>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero vectors
    Eigen::VectorXf v1(2, 1);
    v1 << 1, 4;
    Eigen::VectorXf v2(2, 1);
    v2 << 3, 2;
    Eigen::VectorXf exp(4, 1);
    exp << 3, 2, 12, 8;

    // compute the actual result
    Eigen::VectorXf act = teaser::vectorKron<float, 2, 2>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths
    Eigen::VectorXf v1(3, 1);
    v1 << 1, -3, 3;
    Eigen::VectorXf v2(4, 1);
    v2 << 3, -1, 9, -10;
    Eigen::VectorXf exp(12, 1);
    exp << 3, -1, 9, -10, -9, 3, -27, 30, 9, -3, 27, -30;

    // compute the actual result
    Eigen::VectorXf act = teaser::vectorKron<float, 3, 4>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths (all zero)
    Eigen::VectorXf v1(3, 1);
    v1 << 0, 0, 0;
    Eigen::VectorXf v2(4, 1);
    v2 << 3, -1, 9, -10;
    Eigen::VectorXf exp(12, 1);
    exp.setZero();

    // compute the actual result
    Eigen::VectorXf act = teaser::vectorKron<float, 3, 4>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
}

TEST(LinalgTest, GetNearestPSDRandom) {
  // random psd matrix
  size_t trials = 5;
  int MAX_SIZE = 500;
  int MIN_SIZE = 100;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> size_distribution(MIN_SIZE, MAX_SIZE);

  for (size_t i = 0; i < trials; ++i) {
    int size = size_distribution(generator);
    Eigen::MatrixXf rand_mat = Eigen::MatrixXf::Random(size, size);
    // should be a PSD matrix
    Eigen::MatrixXf A = rand_mat * rand_mat.transpose();
    Eigen::MatrixXf exp_PSD = A;
    Eigen::MatrixXf act_PSD;
    teaser::getNearestPSD<float>(A, &act_PSD);
    ASSERT_TRUE(act_PSD.isApprox(exp_PSD));
  }
}

TEST(LinalgTest, GetNearestPSDSimple) {
  {
    // identity
    Eigen::MatrixXf A = Eigen::MatrixXf::Identity(5, 5);
    Eigen::MatrixXf exp_PSD = A;
    Eigen::MatrixXf act_PSD;
    teaser::getNearestPSD<float>(A, &act_PSD);

    std::cout << "Expected: " << exp_PSD << std::endl;
    std::cout << "Actual: " << act_PSD << std::endl;

    ASSERT_TRUE(act_PSD.isApprox(exp_PSD));
  }
  {
    // another simple example
    Eigen::MatrixXf A(2, 2);
    A << 1, -1, -1, 1;
    Eigen::MatrixXf exp_PSD = A;
    Eigen::MatrixXf act_PSD;
    teaser::getNearestPSD<float>(A, &act_PSD);

    std::cout << "Expected: " << exp_PSD << std::endl;
    std::cout << "Actual: " << act_PSD << std::endl;

    ASSERT_TRUE(act_PSD.isApprox(exp_PSD));
  }
  {
    // a more difficult case
    Eigen::MatrixXf A(5, 5);
    // clang-format off
    A << 4.985750134559297e-02, 6.013987552528092e-01, 6.534568239799645e-01, 2.989641634571971e-01, 9.833730174463773e-02,
         5.458862088908298e-01, 7.896204651230384e-01, 4.896553453284711e-01, 2.561097810618381e-01, 8.595934534752724e-01,
         9.431698396554787e-01, 7.991850350443156e-01, 9.728522370108029e-01, 8.865637944115470e-01, 2.762900816117508e-02,
         3.214730698699890e-01, 4.956476557821099e-02, 7.484899099770820e-01, 4.468008627828005e-01, 8.991564342563644e-01,
         8.064668037933336e-01, 2.831986338409456e-01, 5.678411497388073e-01, 8.159872532962074e-01, 8.999355005261755e-01;
    // clang-format on
    Eigen::MatrixXf exp_PSD(5, 5);
    // clang-format off
    exp_PSD << 4.066470610623304e-01, 5.058983010895730e-01, 6.286981536486869e-01, 3.710481485062439e-01, 3.824610939303472e-01,
               5.058983010895730e-01, 8.624505164355950e-01, 6.061105353450932e-01, 2.624371560274579e-01, 5.037336021002040e-01,
               6.286981536486869e-01, 6.061105353450932e-01, 1.136403337757137e+00, 6.461507810773540e-01, 4.261633102468473e-01,
               3.710481485062439e-01, 2.624371560274579e-01, 6.461507810773540e-01, 7.019256648504395e-01, 6.821231201538008e-01,
               3.824610939303472e-01, 5.037336021002040e-01, 4.261633102468473e-01, 6.821231201538008e-01, 1.022899463404716e+00;
    // clang-format on

    Eigen::MatrixXf act_PSD;
    teaser::getNearestPSD<float>(A, &act_PSD);

    std::cout << "Expected: " << exp_PSD << std::endl;
    std::cout << "Actual: " << act_PSD << std::endl;

    ASSERT_TRUE(act_PSD.isApprox(exp_PSD));
  }
}
