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

#ifndef __LIPNET_NETWORK_PROBLEM_BATCH_L2_HPP__
#define __LIPNET_NETWORK_PROBLEM_BATCH_L2_HPP__

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
     * @brief The network_problem_batch_l2_t struct. The problem implementation of l2 neural
     *        network training in batches.
     *
     *         @f[ \nabla_{W,b} \mathcal{L}(f_{W,b}) + \frac{\rho}{2} ||W||^2 + \frac{\rho}{2} ||b||^2  @f]
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
    struct network_problem_batch_l2_t :
            public backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...>,
            public problem_t< T, problem_type::NONLINEAR, network_problem_batch_l2_t<T,ATYPE,LOSS,N...>,
                typename network_t<T,ATYPE, N...>::layer_t, typename network_t<T,ATYPE, N...>::layer_t > {

        typedef backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...> self_back_t;
        typedef typename self_back_t::variable_t variable_t;


        struct metainfo_t : public self_back_t::metainfo_t {
            using self_back_t::metainfo_t::metainfo_t;
        };


        const T rho = 1.0;

        /**
         * @brief network_problem_batch_l2_t; default constructor
         * @param l loss object
         * @param data traiing data
         * @param rho hyperparameter of L2 regularisation
         */

        explicit network_problem_batch_l2_t(  LOSS<T>&& l,
               network_data_t<T,at<0,N...>(), at<self_back_t::L::value,N...>() > &&data, const T rho = 1.0 )
            : self_back_t( std::move(l), std::move(data) ), rho{ rho } { }


        /**
         * @brief The operator () function. compute gradient
         * @param var Current position
         * @return Gradient and loss at specified position
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info ) const {

            variable_t gradient; T objective = 0;

            // L2 regularisation
            std::for_range<0,self_back_t::L::value>([&]<auto I>(){
                std::get<I>(gradient).weight = 2*rho*std::get<I>(var).weight;
                std::get<I>(gradient).bias = 2*rho*std::get<I>(var).bias;
            });


             // gradient of loss function
            std::invoke( &self_back_t::run, *this, var , info, gradient , objective);

            return std::make_tuple( std::move(gradient) ,
                                    std::move(objective) );
        }

    };

}

#endif // __LIPNET_NETWORK_PROBLEM_BATCH_L2_HPP__
