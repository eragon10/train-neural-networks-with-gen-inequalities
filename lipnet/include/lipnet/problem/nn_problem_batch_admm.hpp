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

#ifndef __LIPNET_NETWORK_PROBLEM_BATCH_ADMM_HPP__
#define __LIPNET_NETWORK_PROBLEM_BATCH_ADMM_HPP__

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

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>
#include <lipnet/network/backpropagation.hpp>

namespace lipnet {



    /**
     * @brief The network_problem_batch_admm_t struct. The problem implementation of admm neural
     *        network training in batches.
     *
     *         @f[ \nabla_{W,b} \mathcal{L}(f_{W,b}) + L_v(W,\tilde{W},y) @f]
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
    struct network_problem_batch_admm_t :
            public backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...>,
            public problem_t< T, problem_type::NONLINEAR, network_problem_batch_admm_t<T,ATYPE,LOSS,N...>,
                typename network_t<T,ATYPE, N...>::layer_t, typename network_t<T,ATYPE, N...>::layer_t > {


        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;


        typedef backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...> self_back_t;
        typedef typename self_back_t::variable_t variable_t;

        struct metainfo_t : public self_back_t::metainfo_t {
            using self_back_t::metainfo_t::metainfo_t;
        };

        /// dual variable
        const variable_t &dualvariable;

        /// weights and biases
        /// variable x
        const variable_t &weights_bar;

        /// admm hyperparameter
        const T rho;

        explicit network_problem_batch_admm_t( LOSS<T>&& l, network_data_t<T, at<0,N...>(), at<L::value,N...>() > &&data,
                                               const T rho,  const variable_t &dualvariable, const variable_t &weights_bar )
            : self_back_t( std::move(l), std::move(data) ) , rho{rho},
              dualvariable{dualvariable}, weights_bar{weights_bar}  {}




        /**
         * @brief The operator () function. compute gradient
         * @param var current position
         * @return gradient and loss at specified position
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info ) const {
            variable_t gradient; T objective = 0;

            std::for_range<0,L::value>([&]<auto I>(){
                std::get<I>(gradient).weight = 0;
                std::get<I>(gradient).bias = 0;
            });


            // compute gradient of loss function
            std::invoke( &self_back_t::run, *this, var, info, gradient, objective);


            // compute augmented lagrange multiplier part
            std::for_range<0,L::value>([&]<auto I>(){
                objective += rho / 2 * blaze::sum( blaze::pow( std::get<I>(var).weight - std::get<I>(weights_bar).weight, 2))
                                           + blaze::sum( blaze::pow( std::get<I>(dualvariable).weight , 2)) ;


                std::get<I>(gradient).weight = std::get<I>(gradient).weight
                      + rho * ( std::get<I>(var).weight - std::get<I>(weights_bar).weight )
                                + std::get<I>(dualvariable).weight;

            });


            return std::make_tuple( std::move(gradient) ,
                                    std::move(objective) );
        }

    };


}

#endif // __LIPNET_NETWORK_PROBLEM_BATCH_ADMM_HPP__
