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

#ifndef __LIPNET_ADAM_BARRIER_HPP__
#define __LIPNET_ADAM_BARRIER_HPP__

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
#include <fstream>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>

#include <lipnet/traits.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/problem.hpp>
#include <lipnet/variable.hpp>
#include <lipnet/statistics.hpp>



namespace lipnet {

    /**
     * @brief Modified adam method for use with barrier functions;
     *        it follows the central path
     * @tparam T numerical value type
     * @tparam P problem type
     * @tparam VAR variable type
     * @tparam GRAD gradient type
     * @tparam feasibility_enabled set this value to true
     *         if you want to enable feasibility checking
     */

    template<typename T, typename P, typename VAR, typename GRAD,
             bool feasibility_enabled = false>
    struct adam_barrier_t_impl
    {

        inline void unpack( std::tuple<GRAD,T> &&t, GRAD &dx, T &fx) const {
            dx = std::move( std::get<0>(t) );
            fx = std::move( std::get<1>(t) );
        }


        static_assert ( helper_function_t<GRAD>::value,
                            "have to provide ´sqrt´ und ´square´");
        

        /**
         * @brief The parameter_t struct; all meta parameters for optimisation
         */

        struct parameter_t {
            size_t max_iter;        /// maximal iterations (default = 5e5)
            size_t cpsteps;         /// central path steps (default = 5)
            T diff;                 /// stopping criterion loss difference (default = 1e-10)
            T threshold;            /// stopping criterion window threshold (default = 1e-8)
            size_t window;          /// stopping criterion window size (default = 300)
            T gamma;                /// barriere factor (default = 1)
            T alpha;                /// stepsize (default = 0.02)
            T beta1;                /// adam meta parameter beta1 (default = 0.9)
            T beta2;                /// adam meta parameter beta2 (default = 0.999)
            T beta3;                /// meta parameter loss difference decrease factor (default = 5.0)

            T alphadec;             /// meta parameter stepsize decrease factor (default = 0.5)
            T gammadec;             /// meta parameter gamma decrease factor (default = 0.5)

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
         *          (size_t) 5e5, (size_t) 5, 1e-10, 1e-8, 300, 1.0, 0.02, 0.9, 0.999, 5.0, 0.5, 0.5, 1e-8
         *
         */
        explicit adam_barrier_t_impl( parameter_t &&param
                        = parameter_t{ (size_t) 5e5, (size_t) 5, 1e-10, 1e-8, 300, 1.0, 0.02, 0.9, 0.999, 5.0, 0.5, 0.5, 1e-8} )
            : param{ std::move(param) } { }


        /**
         * @brief The run method. Implementation of the optimisation algorithm. Modified Adam-method.
         * @tparam stats_enabled enable/disable logging
         * @param prob problem
         * @param x start variable / inital variable / start point
         * @param stats statistics holder
         * @cite kingma2014method
         */
        template<bool stats_enabled = false, bool problem_stats_exists = statistics_helper::stats_type_exists<P>::value>
        inline std::tuple<VAR,T> run( P &prob, VAR&& x, typename std::conditional<stats_enabled,
                               statistics_t, std::void_type >::type &stats ) const {

            metainfo_t<P> info;
            GRAD gradient; feasibility_t<P,T,VAR> step;
            T fx; T fxl = std::numeric_limits<T>::max();

            GRAD momentum, velocity;

            T gamma = param.gamma;
            T alpha = param.alpha;


            for(int j = 0; j < param.cpsteps; j++){

                // get current stopping criterion for each step in central path
                T diff = param.diff*std::pow(param.beta3,param.cpsteps-j);
                T threshold = param.threshold*std::pow(param.beta3,param.cpsteps-j);


                unpack( prob( x, info, step, gamma ) , gradient, fx );
                if constexpr ( stats_enabled )
                        stats.loss << fx;

                // && avglossdecrease < -1e-8
                T avglossdecrease = -10; //&& avglossdecrease < -diff
                size_t i = 0; fxl = std::numeric_limits<T>::max();
                while( (abs(fxl-fx) > diff && i++ < param.max_iter && avglossdecrease < -threshold ) ) {

                    momentum = param.beta1 * momentum + (1-param.beta1)*gradient;
                    velocity = param.beta2 * velocity + (1-param.beta2)*function_t<GRAD>::square(gradient);

                    GRAD t_momentum = (T(1)/(T(1)- std::pow(param.beta1,(T)i) )) * momentum;
                    GRAD t_velocity = (T(1)/(T(1)- std::pow(param.beta2,(T)i) )) * velocity;


                    auto tmp = param.eps + function_t<GRAD>::sqrt(t_velocity);
                    auto direction = t_momentum / tmp;



                    // feasibility check
                    T dalpha = 1;
                    if constexpr ( feasibility_enabled) {
                            step << direction;
                            if (step() < dalpha*alpha) {
                                momentum = GRAD();
                                velocity = GRAD();
                                dalpha = step()/alpha/4;
                            }
                    }

                    x -= alpha * dalpha * direction;

                    fxl = fx;
                    unpack( prob( x, info, step, gamma ) , gradient, fx );
                    if constexpr ( stats_enabled )
                            stats.loss << fx;


                    avglossdecrease = ( (param.window-1)*avglossdecrease + fx - fxl )
                                                / ((T) param.window);


                    if( i % 100 == 0) {
                        std::cout << " => (" << i << ") loss: " << fx  << "\n";
                    }
                }


                // update stepsize and gamma
                gamma *= param.gammadec;
                alpha *= param.alphadec;

            }

            return std::make_tuple( std::move(x),  fx );
        }


    };
    
}

#endif // __LIPNET_ADAM_BARRIER_HPP__
