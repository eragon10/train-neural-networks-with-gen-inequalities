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

#ifndef __LIPNET_ADMM_HPP__
#define __LIPNET_ADMM_HPP__

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
     * @brief Alternating Direction Method of Multipliers. ADMM
     * @cite boyd2011distributed
     * @tparam T numerical value type
     * @tparam P problem type
     * @tparam X first variable type
     * @tparam Z second variable type
     * @tparam DUAL dual variable type
     */

    template<typename T, typename P, typename X, typename Z, typename DUAL>
    struct admm_optimizer_t_impl
    {


        /**
         * @brief compute residual \f$ A x  + Bz - c \f$
         * @cite boyd2011distributed
         * @param prob problem
         * @param x variable
         * @param z variable
         * @return residual
         */
        inline DUAL residual( const P &prob, const X &x, const Z &z) const {
            return std::invoke( &P::residual, prob, x, z);
        }

        /**
         * @brief optimize first subproblem. \f$ \arg \min_x  L_v(x,z^t,y^t) \f$
         * @cite boyd2011distributed
         * @param prob problem
         * @param x variable
         * @param z const variable
         * @param d dual variable
         * @return optimal point x
         */
        inline X optimize1( const P &prob, const X &x, const Z &z, const DUAL &d) const {
            return std::invoke( &P::optimize1, prob, param.rho, x, z, d);
        }

        /**
         * @brief optimize second subproblem. \f$ \arg \min_z  L_v(x^{t+1},z,y^t) \f$
         * @cite boyd2011distributed
         * @param prob problem
         * @param x const variable
         * @param z variable
         * @param d dual variable
         * @return optimal point z
         */
        inline Z optimize2( const P &prob, const X &x, const Z &z, const DUAL &d) const {
            return std::invoke( &P::optimize2, prob, param.rho, x, z, d);
        }

        /**
         * @brief evaluate augmented lagrangian
         * @param prob problem
         * @param x variable
         * @param z variable
         * @return loss/objectiv
         */
        inline T evaluate( const P &prob, const X &x, const Z &z) const {
            return std::invoke( &P::loss, prob, param.rho, x, z);
        }



        struct parameter_t {
            size_t max_iter;    /// max iterations (default = 1e4)
            T rho;              /// admm hyperparameter (augmented lagrange multiplier parameter) (default = 2)
            T eps;              /// numerical offset (default = 1e-8)
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
         *           (size_t) 1e4, 2, 1e-1
         *
         */
        explicit admm_optimizer_t_impl( parameter_t && param = parameter_t{ (size_t) 1e4, 2, 1e-1} )
            : param{ std::move(param) } {}


        /**
         * @brief The run method. Implementation of the optimisation algorithm. Adam-method.
         * @tparam stats_enabled enable/disable logging
         * @param prob problem
         * @param x start variable / inital variable / start point (first variable)
         * @param z start variable / inital variable / start point (second variable)
         * @param stats statistics holder
         * @cite kingma2014method
         */
        template<bool stats_enabled = false>
        std::tuple<X,Z,T> run( P &prob, X&& x, Z&& z, typename std::conditional<stats_enabled,
                               statistics_t, std::void_type >::type &stats  ) const {

            DUAL dualvariable;
            T loss, last = std::numeric_limits<T>::max();

            size_t i = 0;
            while( abs(loss-last) > param.eps && i < param.max_iter ) {
                i++; last = loss;

                // first step -> first subproblem
                x = optimize1( prob, x, z, dualvariable);
                
                // second step -> second subproblem
                z = optimize2( prob, x, z, dualvariable);

                loss = evaluate( prob, x, z);
                if constexpr ( stats_enabled )
                    stats.loss << loss;

                // third step 
                dualvariable += residual( prob, x, z);

                if ( i % 1 == 0) std::cout << "loss: " << loss << "\n";
            }

            return std::make_tuple( std::move(x),  std::move(z),  loss );
        }


    };

}

#endif // __LIPNET_ADMM_HPP__
