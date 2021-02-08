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

#ifndef __LIPNET_NETWORK_PROBLEM_PROJECTION_HPP__
#define __LIPNET_NETWORK_PROBLEM_PROJECTION_HPP__

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

#include <lipnet/problem/nn_problem_batch.hpp>
#include <lipnet/lipschitz/structure.hpp>
#include <lipnet/lipschitz/topology.hpp>

#include <lipnet/extern/mosek_projection_wot.hpp>


namespace lipnet {



    /**
     * @brief The network_problem_projection_t struct. The problem implementation of projected neural
     *        network training in batches.
     *
     *        @f[  \nabla_{W,b} \mathcal{L}(f_{W,b}) @f]
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
    struct network_problem_projection_t :
            public backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...>,
            public problem_t< T, problem_type::NONLINEAR, network_problem_projection_t<T,ATYPE,LOSS,N...> > {


        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> QN;
        typedef std::integral_constant<size_t, (N + ... )
                    - at<0,N...>() - at<L::value,N...>() > TN;
        typedef std::integer_sequence<size_t, N...> DIMS;


        typedef backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...> self_back_t;
        typedef typename self_back_t::variable_t variable_t;


        struct metainfo_t : public self_back_t::metainfo_t {
            using self_back_t::metainfo_t::metainfo_t;
        };


        T lipschitz, tparaminit ;

        /**
         * @brief network_problem_projection_t; default constructor
         * @param l loss object
         * @param data tarining data
         * @param lip lipschitz constant
         * @param tparam T hyperparameter from \f$ \chi(\Psi^2,W) \f$
         */

        explicit network_problem_projection_t( LOSS<T>&& l,
                network_data_t<T, at<0,N...>(), at<L::value,N...>() > &&data,
                          const T &lip = 70.0, const T &tparam = 100.0 )
            : self_back_t( std::move(l) , std::move(data) ),
              lipschitz{ lip }, tparaminit{ tparam }  { }



        /**
         * @brief compute gradient of objectiv function linke nominell training
         * @param var variable
         * @param info metainfo
         * @return gradients
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info ) const {

            variable_t gradient; T objective = 0;

            std::for_range<0,self_back_t::L::value>([&]<auto I>(){
                std::get<I>(gradient).weight = 0;
                std::get<I>(gradient).bias = 0;
            });

            std::invoke( &self_back_t::run, *this, var, info, gradient, objective);

            return std::make_tuple( std::move(gradient) ,
                                    std::move(objective) );
        }





        /**
         * @brief The projection method. Compute projection
         *          @f[  \min || W - \tilde{W} ||^2 \quad \mathrm{s.t} \quad \chi(\Psi^2,W) \succeq 0 @f]
         * @see lipnet::mosek_projection_wot_t
         */

          variable_t projection(variable_t &&var) const {
              auto res = mosek_projection_wot_t<T,N...>::projection(
                                lipschitz, std::move(var), tparaminit );
              return std::move( res );
          }


    };

}

#endif // __LIPNET_NETWORK_PROBLEM_PROJECTION_HPP__
