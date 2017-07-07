/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/
#include <atomic>
#include <utility>
#include <experimental/optional>


#include <gtest/gtest.h>
#include <iostream>

#include "farm.h"
#include "pipeline.h"
#include "common/polymorphic_execution.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

template <typename T>
class farm_test : public ::testing::Test {
public:
  T execution_;
  polymorphic_execution poly_execution_ = 
    make_polymorphic_execution<T>();

  // Variables
  std::atomic<int> output;


  // Vectors
  vector<int> v{};
  vector<int> v2{};
  vector<int> v3{};
  vector<int> w{};

  // entry counter
  int idx_in = 0;
  int idx_out = 0;

  // Invocation counter
  std::atomic<int> invocations_in{0};
  std::atomic<int> invocations_op{0};
  std::atomic<int> invocations_sk{0};

  void setup_empty() {
  }

  void check_empty() {
    ASSERT_EQ(1, invocations_in); // Functor in was invoked once
    ASSERT_EQ(0, invocations_op); // Functor op was not invoked
    ASSERT_EQ(0, invocations_sk); // Functor op was not invoked
  }

  void setup_single() {
    v = vector<int>{42};
    w = vector<int>{99};
  }

  void check_single() {
    ASSERT_EQ(2, invocations_in); // Functor in was invoked once
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(84, this->w[0]);
  }
  void check_single_sink() {
    ASSERT_EQ(2, invocations_in); // Functor in was invoked once
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(1, this->invocations_sk); // one invocation of function sk
    EXPECT_EQ(84, this->w[0]);
  }

  void setup_single_ary() {
    v = vector<int>{11};
    v2 = vector<int>{22};
    v3 = vector<int>{33};
    w = vector<int>{99};
  }

  void check_single_ary() {
    ASSERT_EQ(2, invocations_in); // Functor in was invoked twice
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(66, this->w[0]);
  }

  void check_single_ary_sink() {
    ASSERT_EQ(2, invocations_in); // Functor in was invoked twice
    EXPECT_EQ(1, this->invocations_op); // one invocation of function op
    EXPECT_EQ(1, this->invocations_sk); // one invocation of function sk
    EXPECT_EQ(66, this->w[0]);
  }

  void setup_multiple() {
    v = vector<int>{1,2,3,4,5};
    output = 0;
  }

  void check_multiple() {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(30, this->output);
  }
  
    void check_multiple_sink() {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(5, this->invocations_sk); // five invocations of function sk
    EXPECT_EQ(30, this->output);
  }

  void setup_multiple_ary() {
    v = vector<int>{1,2,3,4,5};
    v2 = vector<int>{2,4,6,8,10};
    v3 = vector<int>{10,10,10,10,10};
    output = 0;
  }

  void check_multiple_ary() {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(95, this->output);
  }
  
  void check_multiple_ary_sink() {
    EXPECT_EQ(6, this->invocations_in); // six invocations of function in
    EXPECT_EQ(5, this->invocations_op); // five invocations of function op
    EXPECT_EQ(5, this->invocations_sk); // five invocations of function sk
    EXPECT_EQ(95, this->output);
  }

};

// Test for execution policies defined in supported_executions.h
TYPED_TEST_CASE(farm_test, executions);

TYPED_TEST(farm_test, static_empty)
{
  this->setup_empty();
  grppi::farm(this->execution_,
    [this]() ->optional<int>{
      this->invocations_in++;
      return {};
    },
    [this](int x) {
      this->invocations_op++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, static_empty_ary)
{
  this->setup_empty();
  grppi::farm(this->execution_,
    [this]() ->optional<tuple<int,int,int>> {
      this->invocations_in++;
      return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, static_single)
{
  this->setup_single();
  grppi::farm(this->execution_,
    [this]() -> optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      this->w[this->idx_out] = x * 2;
      this->idx_out++;
    }
  );
  this->check_single();
}

TYPED_TEST(farm_test, static_single_ary)
{
  this->setup_single_ary();
  grppi::farm(this->execution_,
    [this]() ->optional<tuple<int,int,int>>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      this->w[this->idx_out] = get<0>(x) + get<1>(x) + get<2>(x);
      this->idx_out++;
    }
  );
  this->check_single_ary();
}

TYPED_TEST(farm_test, static_multiple)
{
  this->setup_multiple();
  grppi::farm(this->execution_,
    [this]() ->optional<int>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      this->output += x * 2;
    }
  );
  this->check_multiple();
}

TYPED_TEST(farm_test, static_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::farm(this->execution_,
    [this]() -> optional<tuple<int,int,int>>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      this->output += get<0>(x) + get<1>(x) + get<2>(x);
    }
  );
  this->check_multiple_ary();
}


TYPED_TEST(farm_test, static_empty_sink)
{
  this->setup_empty();
  grppi::farm(this->execution_,
    [this]() -> optional<int>{
      this->invocations_in++;
      return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x;
    },
    [this](int x) {
      this->invocations_sk++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, static_empty_ary_sink)
{
  this->setup_empty();
  grppi::farm(this->execution_,
    [this]() -> optional<tuple<int,int,int>>{
      this->invocations_in++;
      return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return x;
    },
    [this](tuple<int,int,int> x) {
      this->invocations_sk++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, static_single_sink)
{
  this->setup_single();
  grppi::farm(this->execution_,
    [this]() ->optional<int>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x*2;
    },
    [this](int x) {
      this->invocations_sk++;
      this->w[this->idx_out] = x;
      this->idx_out++;
    }
  );
  this->check_single_sink();
}

TYPED_TEST(farm_test, static_single_ary_sink)
{
  this->setup_single_ary();
  grppi::farm(this->execution_,
    [this]() -> optional<tuple<int,int,int>>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return get<0>(x) + get<1>(x) + get<2>(x);;
    },
    [this](int x) {
      this->invocations_sk++;
      this->w[this->idx_out] = x;
      this->idx_out++;
    }
  );
  this->check_single_ary_sink();
}

TYPED_TEST(farm_test, static_multiple_sink)
{
  this->setup_multiple();
  grppi::farm(this->execution_,
    [this]() ->optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x*2;
    },
    [this](int x) {
      this->invocations_sk++;
      this->output += x;
    }
  );
  this->check_multiple_sink();
}

TYPED_TEST(farm_test, static_multiple_ary_sink)
{
  this->setup_multiple_ary();
  grppi::farm(this->execution_,
    [this]() -> optional<tuple<int,int,int>>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return get<0>(x) + get<1>(x) + get<2>(x);;
    },
    [this](int x) {
      this->invocations_sk++;
      this->output += x;
    }
  );
  this->check_multiple_ary_sink();
}


TYPED_TEST(farm_test, static_empty_composed)
{
  this->setup_empty();
    grppi::pipeline( this->execution_,
    [this]() -> optional<int>{ 
      this->invocations_in++;
      return {};
    },
    grppi::farm(this->execution_,
    [this](int x) {
      this->invocations_op++;
      return x;
    }
    ),
    [this](auto y ) {
      this->invocations_sk++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, static_single_composed)
{
  this->setup_single();
    grppi::pipeline( this->execution_,
    [this]() ->optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    grppi::farm(this->execution_,
    [this](int x) {
      this->invocations_op++;
      return x*2;
    }
    ),
    [this](auto x ) {
      this->invocations_sk++;
      this->w[this->idx_out] = x;
      this->idx_out++;
    }
  );
  this->check_single_sink();
}

TYPED_TEST(farm_test, static_multiple_composed)
{
  this->setup_multiple();
    grppi::pipeline( this->execution_,
    [this]() -> optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    grppi::farm(this->execution_,
    [this](int x) {
      this->invocations_op++;
      return x*2;
    }
    ),
    [this](auto x ) {
      this->invocations_sk++;
      this->output += x;
    }
  );
  this->check_multiple_sink();
}


TYPED_TEST(farm_test, poly_empty)
{
  this->setup_empty();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<int>{
      this->invocations_in++;
      return {};
    },
    [this](int x) {
      this->invocations_op++;
    }
  );
  this->check_empty();
}



TYPED_TEST(farm_test, poly_empty_sink)
{
  this->setup_empty();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<int> {
      this->invocations_in++;
      return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x;
    },
    [this](int x) {
      this->invocations_sk++;
    }

  );
  this->check_empty();
}

TYPED_TEST(farm_test, poly_empty_ary)
{
  this->setup_empty();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<tuple<int,int,int>> {
      this->invocations_in++;
      return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, poly_empty_ary_sink)
{
  this->setup_empty();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<tuple<int,int,int>>{
      this->invocations_in++;
      return {};
    },
  [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return x;
    },
    [this](tuple<int,int,int> x) {
      this->invocations_sk++;
    }
  );
  this->check_empty();
}

TYPED_TEST(farm_test, poly_single)
{
  this->setup_single();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      this->w[this->idx_out] = this->v[this->idx_out] * 2;
      this->idx_out++;
    }
  );
  this->check_single();
}

TYPED_TEST(farm_test, poly_single_sink)
{
  this->setup_single();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<int>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x*2;
    },
    [this](int x) {
      this->invocations_sk++;
      this->w[this->idx_out] = x;
      this->idx_out++;
    }
  );
  this->check_single_sink();
}

TYPED_TEST(farm_test, poly_single_ary)
{
  this->setup_single_ary();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<tuple<int,int,int>> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      this->w[this->idx_out] = get<0>(x) + get<1>(x) + get<2>(x);
      this->idx_out++;
    }
  );
  this->check_single_ary();
}

TYPED_TEST(farm_test, poly_single_ary_sink)
{
  this->setup_single_ary();
  grppi::farm(this->poly_execution_,
    [this]() -> optional<tuple<int,int,int>> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return get<0>(x) + get<1>(x) + get<2>(x);;
    },
    [this](int x) {
      this->invocations_sk++;
      this->w[this->idx_out] = x;
      this->idx_out++;
    }
  );
  this->check_single_ary_sink();
}

TYPED_TEST(farm_test, poly_multiple)
{
  this->setup_multiple();
  grppi::farm(this->poly_execution_,
    [this]() -> optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      this->output += x * 2;
    }
  );
  this->check_multiple();
}

TYPED_TEST(farm_test, poly_multiple_sink)
{
  this->setup_multiple();
  grppi::farm(this->poly_execution_,
    [this]() ->optional<int> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return this->v[this->idx_in-1];
      } else
        return {};
    },
    [this](int x) {
      this->invocations_op++;
      return x*2;
    },
    [this](int x) {
      this->invocations_sk++;
      this->output += x;
    }
  );
  this->check_multiple_sink();
}

TYPED_TEST(farm_test, poly_multiple_ary)
{
  this->setup_multiple_ary();
  grppi::farm(this->poly_execution_,
    [this]() -> optional<tuple<int,int,int>>{
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      this->output += get<0>(x) + get<1>(x) + get<2>(x);
    }
  );
  this->check_multiple_ary();
}

TYPED_TEST(farm_test, poly_multiple_ary_sink)
{
  this->setup_multiple_ary();
  grppi::farm(this->poly_execution_,
    [this]() -> optional<tuple<int,int,int>> {
      this->invocations_in++;
      if ( this->idx_in < this->v.size() ) {
        this->idx_in++;
        return make_tuple(this->v[this->idx_in-1],
                       this->v2[this->idx_in-1],
                       this->v3[this->idx_in-1]);
      } else
        return {};
    },
    [this](tuple<int,int,int> x) {
      this->invocations_op++;
      return get<0>(x) + get<1>(x) + get<2>(x);;
    },
    [this](int x) {
      this->invocations_sk++;
      this->output += x;
    }
  );
  this->check_multiple_ary_sink();
}