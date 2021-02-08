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

#ifndef __LIPNET_NETWORK_PROBLEM_LIPTRAIN_ENFORCING_HPP__
#define __LIPNET_NETWORK_PROBLEM_LIPTRAIN_ENFORCING_HPP__

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
#include <lipnet/tensor.hpp>
#include <lipnet/problem.hpp>

#include <lipnet/optimizer.hpp>

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>

#include <lipnet/problem/nn_problem_batch_admm.hpp>

#include <lipnet/extern/nn_lipcalc.hpp>
#include <lipnet/extern/nn_liptrain_enforcing.hpp>




namespace lipnet {



    /**
     * @brief The network_problem_liptrain_enforcing_adam_t struct. The problem implementation of admm neural
     *        network training to enforce lipschitz bound.
     *
     * @tparam T Base numeric type (eg. double, float, ...).
     * @tparam ATPYE Activation type of this neural network.
     * @tparam LOSS Objectiv function type of this neural network
     * @tparam BATCH Const integer value specifying the batch size.
     * @tparam N Neural network topology. Array of postive integer values specifying the
     *         number of neurons at each layer.
     *
     */

    template<typename T, template<typename> typename ATYPE,
             template<typename> typename LOSS, size_t BATCH, size_t ...N>
    struct network_problem_liptrain_enforcing_adam_t : public problem_t<T, problem_type::ADMM,
            network_problem_liptrain_enforcing_adam_t<T, ATYPE, LOSS, N...>,
            typename network_t<T,ATYPE, N...>::layer_t,
            typename network_t<T,ATYPE, N...>::layer_t,
            typename network_t<T,ATYPE, N...>::layer_t> {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;


        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;


        typedef typename network_t<T,ATYPE, N...>::layer_t variable_t;


        network_data_t<T, at<0,N...>(), at<L::value,N...>() > training_data;
        const T lipschitz;

        /**
         * @brief network_problem_liptrain_enforcing_adam_t; default constructor
         * @param data training data
         * @param lip lipschitz constant
         */

        explicit network_problem_liptrain_enforcing_adam_t( const network_data_t<T,
                              at<0,N...>(), at<L::value,N...>() > &&data, const T lip = 70.0 )

            : training_data { std::move(data) },
              lipschitz{ std::move(lip) } {  }

        /**
         * @brief The residual method; compute residual
         * @param x variable
         * @param z variable
         * @return residual
         */

        variable_t residual( const variable_t &x, const variable_t &z ) const {
            return x - z;
        }


        /**
         * @brief optimize first subproblem; with nominell training; adam method
         *
         *          @f[ \arg \min_{W,b} L_v(W,b,\tilde{W},Y) @f]
         *
         * @param rho admm hyperparameter; augmented lagrange multiplier
         * @param var variable to optimize
         * @param varbar second const variable
         * @param dvar dual variable
         * @return optimal point var
         */

        variable_t optimize1( const T rho, const variable_t &var,
                              const variable_t &varbar, const variable_t &dvar) const {

            typedef network_problem_batch_admm_t<T, ATYPE, LOSS, BATCH, N...> problem_t ;

            network_data_t<T, at<0,N...>(), at<L::value,N...>() > dd = training_data;
            problem_t prob( LOSS<T>{} , std::move(dd), rho, dvar, varbar );

            typedef adam_momentum_t<T, problem_t, typename problem_t::variable_t,
                            typename problem_t::variable_t> subsolver_t;

            subsolver_t solver(  typename subsolver_t::parameter_t{
                 (size_t) 5e3, 1e-6, 1e-4, 0.02, 0.9, 0.999, 1e-8 } );

            typename problem_t::variable_t init = var;
            auto [ weights, l ] = solver( prob, std::move( init ) );

            return std::move( weights );
        }


        /**
         * @brief optimize second variable; conic programm; mosek; interior point method
         *
         *              @f[ \arg \min_{\tilde{W}} L_v(W,b,\tilde{W},Y) @f]
         *
         * @param rho admm hyperparameter; augmented lagrange multiplier
         * @param var first const variable
         * @param varbar variable
         * @param dvar dual variable
         * @return optimal point varvar
         */


        variable_t optimize2( const T rho, const variable_t &var,
                              const variable_t &varbar, const variable_t &dvar) const {

            auto [ lip, tparam ] = network_libcalc_t<T, N...>::solve( var );
            auto weights =  network_libtrain_enforcing_t<T, N...>::train( lipschitz, rho, var,
                    blaze::uniform( sum<L::value,N...>()-at<0,N...>() , 1e4 ) , dvar );
            auto [ k2, mmmm ] = network_libcalc_t<T, N...>::solve( weights );

            std::cout << " =================> LIPSCHITZ: real: " << lip << " <-> sub: " << k2 << "\n";
            return std::move( weights );
        }


        /**
         * @brief compute lipschitz constant; mosek; interior point method;
         * @param rho admm hyperparameter; augmented lagrange multiplier
         * @param var first const variable
         * @param varbar second const variable
         * @return lipschitz constant
         */

        T loss( const T rho, const variable_t &var, const variable_t &varbar ) const {
            auto [ lip, tparam ] = network_libcalc_t<T, N...>::solve( var );
            return lip;
        }



    };

}

#endif // __LIPNET_NETWORK_PROBLEM_LIPTRAIN_ENFORCING_HPP__
