/*
 * Copyright 2020 Niklas Funcke <niklas.funcke@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LIPNET_PROBLEM_HPP__
#define __LIPNET_PROBLEM_HPP__

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <list>
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <initializer_list>
#include <deque>

#include <lipnet/traits.hpp>
#include <lipnet/tuple.hpp>
#include <lipnet/tensor.hpp>



namespace lipnet {
    

    /**
     * @brief The problem_type enum; all possible problem class
     */

    enum problem_type { SDP, NONLINEAR, LINEAR, QUADRATIC, ADMM, AUGLAG, ITERATION };



    /**
     * @brief The problem_t struct; base problem struct (basically a placerholder class)
     * @tparam T numerical value type
     * @tparam TYPE problem class
     * @tparam IMPL actual problem struct
     * @tparam ARGS problem specific types (passthrough)
     */

    template<typename T, problem_type TYPE, typename IMPL, typename ...ARGS>
    struct problem_t {};


    /**
     * @brief The metainfo_t struct. Data holder type for data needed during the iterations.
     * @tparam IMPL problem type
     */

    template<typename IMPL>
    struct metainfo_t : public IMPL::metainfo_t {
        explicit metainfo_t() : IMPL::metainfo_t() {}
    };


    /**
     * @brief The linesearch_t struct. base linesearch struct (basically a placerholder class)
     * @tparam IMPL problem type
     * @tparam T numerical value type
     * @tparam DIRECTION variable type
     */

    template<typename IMPL, typename T, typename DIRECTION>
    struct linesearch_t : public IMPL::linesearch_t {

        /// @brief evaluate function with stepsize val.
        /// @param val stepsize @f[ \alpha \quad x_{k+1} = x_{k} - \alpha \Delta x @f]
        inline T operator()(const T val) const {
            return std::invoke( &IMPL::linesearch_t::run, *this, val );
        }

        /// @brief set direction for evaluation.
        /// @param dir direction @f[ \Delta x \quad x_{k+1} = x_{k} - \alpha \Delta x @f]
        inline void operator<<(const DIRECTION& dir) {
            std::invoke( &IMPL::linesearch_t::update, *this, dir );
        }
    };


    /**
     * @brief The feasibility_t struct. base feasibility struct (basically a placerholder class)
     * @tparam IMPL problem type
     * @tparam T numerical value type
     * @tparam DIR variable type
     */

    template<typename IMPL, typename T, typename DIR>
    struct feasibility_t : public IMPL::feasibility_t {

        /// @brief compute max stepsize for problem specific constraint.
        ///        @f[ \hat{\alpha} = \max_\alpha \alpha  \quad \mathrm{s.t.} \quad  [x_{k} - \alpha \Delta x] \mathrm{\; is\; feasible} @f]
        inline T operator()() const {
            return std::invoke( &IMPL::feasibility_t::step, *this );
        }

        /// @brief set direction for evaluation.
        /// @param dir direction @f[ \Delta x \quad x_{k+1} = x_{k} - \alpha \Delta x @f]
        inline void operator<<( const DIR& dir ) {
            std::invoke( &IMPL::feasibility_t::run, *this, dir );
        }
    };

}

#endif // __LIPNET_PROBLEM_HPP__
