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

#ifndef __LIPNET_ADAM_HPP__
#define __LIPNET_ADAM_HPP__

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
     * @brief The Adam method.
     * @cite kingma2014method
     * @tparam T numerical value type
     * @tparam P problem type
     * @tparam VAR variable type
     * @tparam GRAD gradient type
     */

    template<typename T, typename P, typename VAR, typename GRAD>
    struct adam_momentum_t_impl
    {

        typedef std::function<bool (const T&, const VAR&, const GRAD&)> criterion_t;

        inline void unpack( std::tuple<GRAD,T> &&t, GRAD &dx, T &fx) const {
            dx = std::move( std::get<0>(t) );
            fx = std::move( std::get<1>(t) );
        }


        static_assert ( helper_norm_t<T, GRAD>::value,
                            "have to provide ´norm´");
        static_assert ( helper_function_t<GRAD>::value,
                            "have to provide ´sqrt´ und ´square´");
        

        struct parameter_t {
            size_t max_iter;        /// max iterations (default = 5e5)
            T diff;                 /// stopping criterion loss difference (default = 1e-10)
            T graddiff;             /// stopping criterion gradient norm (default = 1e-4)
            T alpha;                /// stepsize (default = 0.02)
            T beta1;                /// adam meta parameter beta1 (default = 0.9)
            T beta2;                /// adam meta parameter beta2 (default = 0.999)
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
        /// custom stopping criterion
        criterion_t criterion;

        /**
         * @brief Default constructor.
         * @param hyperparameter of optimisation. Init hyperparameters with
         *          (size_t) 5e4, 1e-10, 1e-4, 0.02, 0.9, 0.999, 1e-8
         *
         */
        explicit adam_momentum_t_impl( parameter_t &&param
                        = parameter_t{ (size_t) 5e4, 1e-10, 1e-4, 0.02, 0.9, 0.999, 1e-8},
          criterion_t &&c = [](const T&,const VAR&,const GRAD&){return true;} )
            : param{ std::move(param) }, criterion{ std::move(c) }  { }


        /**
         * @brief The run method. Implementation of the optimisation algorithm. Adam-method.
         * @tparam stats_enabled enable/disable logging
         * @param prob problem
         * @param x start variable / inital variable / start point
         * @param stats statistics holder
         * @cite kingma2014method
         */
        template<bool stats_enabled = false>
        inline std::tuple<VAR,T> run( P &prob, VAR&& x, typename std::conditional<stats_enabled,
                               statistics_t, std::void_type >::type &stats ) const {

            GRAD gradient; metainfo_t<P> info;
            T fx; T fxl = std::numeric_limits<T>::max();

            GRAD momentum, velocity;

            unpack( prob( x, info ) , gradient, fx );
            std::cout << "START => loss: " << fx
                      << "     -- norm: "
                      << norm_t<T,GRAD>::norm(gradient) << "\n";

            //std::cout << std::get<0>(gradient).weight << "\n";
            //std::cout << std::get<1>(gradient).weight << "\n";
            //std::cout << std::get<>(gradient).weight << "\n";

            if constexpr ( stats_enabled )
                    stats.loss << fx;

            size_t i = 0; // norm_t<T,GRAD>::norm(gradient) > eps    //  (fxl-fx) > eps
            while( (abs(fxl-fx) > param.diff && norm_t<T,GRAD>::norm(gradient) > param.graddiff
                    && i++ < param.max_iter) && criterion(fx,x,gradient) ) {

                momentum = param.beta1 * momentum + (1-param.beta1)*gradient;
                velocity = param.beta2 * velocity + (1-param.beta2)*function_t<GRAD>::square(gradient);

                GRAD t_momentum = (T(1)/(T(1)- std::pow(param.beta1,(T)i) )) * momentum;
                GRAD t_velocity = (T(1)/(T(1)- std::pow(param.beta2,(T)i) )) * velocity;


                auto tmp = param.eps + function_t<GRAD>::sqrt(t_velocity);
                x -= param.alpha * t_momentum / tmp;


                fxl = fx;
                unpack( prob( x, info ) , gradient, fx );

                if constexpr ( stats_enabled )
                        stats.loss << fx;

                if( i % 100 == 0) {
                    std::cout << " => (" << i << ") loss: "
                              << fx << "     -- norm: "
                              << norm_t<T,GRAD>::norm(gradient) << "\n";
                }
            }


            unpack( prob( x, info ) , gradient, fx );
            std::cout << "END => loss: " << fx
                      << "     -- norm: "
                      << norm_t<T,GRAD>::norm(gradient) << "\n";

            return std::make_tuple( std::move(x),  fx );
        }


    };
    
}

#endif // __LIPNET_ADAM_HPP__
