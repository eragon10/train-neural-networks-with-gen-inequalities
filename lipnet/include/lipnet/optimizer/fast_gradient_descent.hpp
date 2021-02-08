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

#ifndef __LIPNET_FAST_GRADIENT_DESCENT_HPP__
#define __LIPNET_FAST_GRADIENT_DESCENT_HPP__

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
#include <iostream>

#include <lipnet/traits.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/problem.hpp>
#include <lipnet/variable.hpp>
#include <lipnet/statistics.hpp>




namespace lipnet {

    /**
     * @brief gradient descent algorithm.
     * @tparam T numerical value type
     * @tparam P problem type
     * @tparam VAR variable type
     * @tparam GRAD gradient type
     */
    template<typename T, typename P, typename VAR, typename GRAD>
    struct fast_gradient_descent_t_impl
    {

        inline void unpack( std::tuple<GRAD,T> &&t, GRAD &dx, T &fx) const {
            dx = std::move( std::get<0>(t) );
            fx = std::move( std::get<1>(t) );
        }


        static_assert ( helper_norm_t<T, GRAD>::value,
                            "have to provide ´norm´");



        struct parameter_t {
            T gamma;                /// stepsize (default = 0.001)
            T eps;                  /// numerical offset (default = 1e-8)
        };


        /// @brief problem specific implementation of statistics_t
        /// @see lipnet statistics_t
        /// @cite cereallib
        struct statistics_t {
            series_t<T> loss;

            template<class Archive> void serialize(Archive & archive)
                {  archive( cereal::make_nvp("loss", loss) ); }
        };


        /// variables to optimize
        parameter_t param;

        /**
         * @brief Default constructor.
         * @param hyperparameter of optimisation. Init hyperparameters with
         *           0.001, 1e-8
         *
         */
        explicit fast_gradient_descent_t_impl(parameter_t &&param = parameter_t{0.001, 1e-8})
            : param{ std::move(param) } {}


        /**
         * @brief The run method. Implementation of the optimisation algorithm.
         * @tparam stats_enabled enable/disable logging
         * @param prob problem
         * @param x start variable / inital variable / start point
         * @param stats statistics holder
         */
        template<bool stats_enabled = false>
        inline std::tuple<VAR,T> run( P &prob, VAR&& x, typename std::conditional<stats_enabled,
                               statistics_t, std::void_type >::type &stats ) const {

            metainfo_t<P> info;

            GRAD gradient; T fx;
            unpack( prob( x, info ) , gradient, fx );
            if constexpr ( stats_enabled )
                    stats.loss << fx;

            size_t i = 0;
            while( norm_t<T,GRAD>::norm(gradient) > param.eps ) {
                i++;

                x -= param.gamma * gradient;
                unpack( prob( x, info ) , gradient, fx );
                if constexpr ( stats_enabled )
                        stats.loss << fx;

                if( i % 1000 == 0) {
                    std::cout << " => (" << i << ") loss: "
                              << fx << "     -- norm: "
                              << norm_t<T,GRAD>::norm(gradient) << "\n";
                }
            }

            return std::make_tuple( std::move(x),  fx );
        }




    };

}

#endif // __LIPNET_FAST_GRADIENT_DESCENT_HPP__
