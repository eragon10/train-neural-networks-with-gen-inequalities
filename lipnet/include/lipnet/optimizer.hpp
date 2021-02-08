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

#ifndef __LIPNET_OPTIMIZER_HPP__
#define __LIPNET_OPTIMIZER_HPP__

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
#include <optional>
#include <chrono>

#include <lipnet/traits.hpp>
#include <lipnet/tuple.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/problem.hpp>
#include <lipnet/statistics.hpp>

//#include <lipnet/optimizer/gradient_descent.hpp>
#include <lipnet/optimizer/fast_gradient_descent.hpp>
#include <lipnet/optimizer/gradient_descent_projected.hpp>

//#include <lipnet/optimizer/newton_method.hpp>
//#include <lipnet/optimizer/bfgs_algorithm.hpp>
//#include <lipnet/optimizer/iteration_criterion.hpp>

#include <lipnet/optimizer/adam_momentum.hpp>
#include <lipnet/optimizer/adam_barrier.hpp>
#include <lipnet/optimizer/adam_projected.hpp>

#include <lipnet/optimizer/admm_optimizer.hpp>
//#include <lipnet/optimizer/augmented_lagrangian.hpp>



namespace lipnet {



    template<typename P, typename VAR>
    struct solve_function_helper {
        template <typename T>
        using member_solve_t = decltype(std::declval<T>().solve(
            std::declval<const VAR&>() ));

        inline constexpr static bool value = std::is_detected<member_solve_t, P>::value;
    };


    /**
     * @brief The optimizer_t struct. On instantiation a class with the implementation
     * as base class will be created.
     *
     * @tparam T The numeric base type (e.g. double, float, ...)
     * @tparam P The problem struct, which should be solved (e.g. lasso_problem, ...)
     * @tparam IMPL The implementation of the solver, which should be used
     * @tparam VARS Parameterpack of all type the implementation needs to solve
     *         the problem (e.g. VAR, GRADIENT, DUAL, ...)
     */

    template<typename T, typename P, typename IMPL, typename ...VARS> // typename VAR
    struct optimizer_t : public IMPL
    {

        //static_assert ( std::is_base_of<problem_t<P, T, VAR>, P>::value,
        //        "problem have to be derivived form ´problem´");


        /**
         * @brief Type holder.
         */

        template<class TT> struct void_t { typedef void type; };

        template<class TT, class U = void>
        struct stats_type_exists { enum { value = 0 }; typedef std::void_type type; };

        template<class TT>
        struct stats_type_exists<TT, typename
                void_t<typename TT::statistics_t>::type> { enum { value = 1 };
                   typedef typename TT::statistics_t type; };


        static_assert ( stats_type_exists<IMPL>::value,
                            "solver have to provide ´statistics_t´ type");

        typedef typename stats_type_exists<P>::type statistics_problem_t;

        /**
         * @brief The main_statistics_t struct.
         * @details Just contains a variable to, which stores th computation time to solve
         *          the problem. The variable stores it's value in milliseconds.
         */

        struct main_statistics_t : public IMPL::statistics_t, public statistics_problem_t  {
            std::chrono::milliseconds duration;

            template<class Archive> void serialize(Archive & archive)
                {   static_cast<typename IMPL::statistics_t>(*this).serialize( archive );
                    archive( cereal::make_nvp("optimization-time",  duration.count() ) ); }
        };


        /**
         * @brief The mainoptimization function.
         * @tparam stats_enabled Boolean value to decide if you want to create a statistic about
         *         this optimization process.
         * @param prob The problem variable
         * @param vars The initial values over which you want to optimize
         * @param stats The statistics struct if you want to create statistics or just a void_type if not.
         * @return Optimal value and optimal loss
         */

        template<bool stats_enabled = false>
        inline std::tuple<VARS...,T> run( P &prob, VARS&& ...vars ,  typename std::conditional<stats_enabled,
                                          main_statistics_t, std::void_type >::type &stats ) const {

            auto t1 = std::chrono::high_resolution_clock::now();
            auto res = std::invoke( &IMPL::template run<stats_enabled>, *this,
                                    prob, std::forward<VARS>(vars)..., stats );
            auto t2 = std::chrono::high_resolution_clock::now();

            if constexpr (stats_enabled)
                stats.duration = std::chrono::duration_cast<
                                    std::chrono::milliseconds>(t2-t1);

            std::cout << "\n\n ===> duration: " << std::chrono::duration_cast<
                         std::chrono::milliseconds>(t2-t1).count() << "\n\n";

            return std::move(res);
        }


        /**
         * @brief The operator() function. A wrapper for run(P &prob, VARS&& ...vars ,
         *        typename std::conditional<stats_enabled,   main_statistics_t, std::void_type >::type &stats)
         *        with statistics enabled.
         * @param prob
         * @param vars
         * @param stats
         * @return Optimal value and optimal loss
         * @see run( P &prob, VARS&& ...vars ,  typename std::conditional<stats_enabled,
         *      main_statistics_t, std::void_type >::type &stats )
         */

        std::tuple<VARS...,T> operator()( P &prob, VARS&& ...vars, main_statistics_t &stats ) const {
            return run<true>( prob, std::forward<VARS>(vars) ..., stats);
        }

        /**
         * @brief The operator() function. A wrapper for run(P &prob, VARS&& ...vars ,
         *        typename std::conditional<stats_enabled,   main_statistics_t, std::void_type >::type &stats)
         *        with statistics disabled
         * @param prob
         * @param vars
         * @param stats
         * @return Optimal value and optimal loss
         * @see run( P &prob, VARS&& ...vars ,  typename std::conditional<stats_enabled,
         *      main_statistics_t, std::void_type >::type &stats )
         */

        std::tuple<VARS...,T> operator()( P &prob, VARS&& ...vars ) const {
            std::void_type void_obj;
            return run<false>( prob, std::forward<VARS>(vars) ..., void_obj);
        }

        using IMPL::IMPL;
    };









    //template<typename T, typename P, typename VAR, typename GRAD>
    //using gradient_descent_t = optimizer_t<T, P, gradient_descent_t_impl<T, P, VAR, GRAD>, VAR>;

    template<typename T, typename P, typename VAR, typename GRAD>
    using fast_gradient_descent_t = optimizer_t<T, P, fast_gradient_descent_t_impl<T, P, VAR, GRAD>, VAR>;

    template<typename T, typename P, typename VAR, typename GRAD>
    using gradient_descent_projected_t = optimizer_t<T, P, gradient_descent_projected_t_impl<T, P, VAR, GRAD>, VAR>;


    template<typename T, typename P, typename VAR, typename GRAD>
    using adam_momentum_t = optimizer_t<T, P, adam_momentum_t_impl<T, P, VAR, GRAD>, VAR>;

    template<typename T, typename P, typename VAR, typename GRAD, bool feasibility_enabled = false>
    using adam_barrier_t = optimizer_t<T, P, adam_barrier_t_impl<T, P, VAR, GRAD, feasibility_enabled>, VAR>;

    template<typename T, typename P, typename VAR, typename GRAD>
    using adam_projected_t = optimizer_t<T, P, adam_projected_t_impl<T, P, VAR, GRAD>, VAR>;


    //template<typename T, typename P, typename VAR, typename GRAD, typename HESS>
    //using newton_method_t = optimizer_t<T, P, newton_method_t_impl<T, P, VAR, GRAD, HESS>, VAR>;

    //template<typename T, typename P, typename VAR, typename GRAD, typename HESS>
    //using bfgs_algorithm_t = optimizer_t<T, P, bfgs_algorithm_t_impl<T, P, VAR, GRAD, HESS>, VAR>;

    //template<typename T, typename P, typename VAR>
    //using iteration_algorithm_t = optimizer_t<T, P, iteration_criterion_t_impl<T, P, VAR>, VAR>;



    template<typename T, typename P, typename X, typename Z, typename DUAL>
    using admm_optimizer_t = optimizer_t<T, P, admm_optimizer_t_impl<T, P, X, Z, DUAL>, X, Z>;

    //template<typename T, typename P, typename VAR, typename DUAL>
    //using augmented_lagrangian_t = optimizer_t<T, P, augmented_lagrangian_t_impl<T, P, VAR, DUAL>, VAR>;

}

#endif // __LIPNET_OPTIMIZER_HPP__
