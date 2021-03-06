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

#ifndef __LIPNET_NETWORK_PROBLEM_LOG_BARRIER_WOT_HPP__
#define __LIPNET_NETWORK_PROBLEM_LOG_BARRIER_WOT_HPP__

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

#include <lipnet/traits.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/problem.hpp>

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>
#include <lipnet/network/backpropagation.hpp>

#include <lipnet/lipschitz/topology.hpp>
#include <lipnet/lipschitz/barrier_wot.hpp>
#include <lipnet/lipschitz/feasibility.hpp>



namespace lipnet {


    /**
     * @brief The network_problem_log_barrier_wot_t struct. The problem implementation of barrier (without T) neural
     *        network training in batches.
     *
     *        @f[ \nabla_{W,b} \mathcal{L}(f_{W,b}) - \rho \log \det ( \chi(\Psi^2,W) ) @f]
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
    struct network_problem_log_barrier_wot_t :
            public backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...>,
            public barrierfunction_wot_t<T,N...>,
            public problem_t< T, problem_type::NONLINEAR,
                network_problem_log_barrier_wot_t<T,ATYPE,LOSS,N...>,
                typename network_t<T,ATYPE, N...>::layer_t, typename network_t<T,ATYPE, N...>::layer_t > {


        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integral_constant<size_t, (N + ... )
                    - at<0,N...>() - at<L::value,N...>() > TN;



        typedef backpropagation_batch_t<T, ATYPE, LOSS, BATCH, N...> self_back_t;
        typedef barrierfunction_wot_t<T, N...> self_barrier_t;

        typedef typename self_barrier_t::tparam_t param_t;
        typedef typename self_back_t::variable_t variable_t;

        struct metainfo_t : public self_back_t::metainfo_t {
            using self_back_t::metainfo_t::metainfo_t;
        };


        /**
         * @brief The feasibility_t struct. Implementation of feasibility check
         *          for this problem
         * @see lipnet::feasibilitycheck_wot_t
         */

        struct feasibility_t : public feasibilitycheck_wot_t<T,N...> {
            typename self_barrier_t::cholesky_t L; T step;
            std::optional<std::reference_wrapper<const typename self_barrier_t::tparam_t>> Tparam;

            void init( typename self_barrier_t::cholesky_t &&l,
                       const typename self_barrier_t::tparam_t &t) {
                L = std::move(l); Tparam = std::cref(t);
            }


            void run( const variable_t& dir ) {
                step = std::invoke( &feasibilitycheck_wot_t<T,N...>::compute,
                                    *this, Tparam.value().get(), L, dir );
            }
        };


        /**
         * @brief network_problem_log_barrier_wot_t; default constructor
         * @param l loss object
         * @param data training data
         * @param tparam T hyperparameter from \f$ \chi(\Psi^2,W) \f$
         * @param lipschitz lipschitz constant
         */

        explicit network_problem_log_barrier_wot_t(
                LOSS<T>&& l, network_data_t<T,at<0,N...>(), at<L::value,N...>() > &&data,
                param_t&& tparam,  const T lipschitz = 70.0 )
            : self_back_t( std::move(l), std::move(data) ),
              self_barrier_t( std::move(tparam) , lipschitz ){ }


        /**
         * @brief compute gradients
         * @param var variable
         * @param info metainfo
         * @param line feasibility check
         * @param gamma hyperparameter
         * @return gradients
         * @see run( const variable_t& var, metainfo_t &info,
         *        typename std::conditional<feasibility_enabled, feasibility_t, std::void_type >::type &feasibility,
         *        typename std::conditional<gamma_enabled, T, std::void_type >::type level ) const
         *
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info,
                                             feasibility_t &line, T &gamma) const {
            return run<true, true>( var, info, line, gamma );
        }

        /**
         * @see run( const variable_t& var, metainfo_t &info,
         *        typename std::conditional<feasibility_enabled, feasibility_t, std::void_type >::type &feasibility,
         *        typename std::conditional<gamma_enabled, T, std::void_type >::type level ) const
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info,
                                             feasibility_t &line ) const {
            std::void_type void_obj;
            return run<true, false>( var, info, line, void_obj );
        }

        /**
         * @see run( const variable_t& var, metainfo_t &info,
         *        typename std::conditional<feasibility_enabled, feasibility_t, std::void_type >::type &feasibility,
         *        typename std::conditional<gamma_enabled, T, std::void_type >::type level ) const
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info,
                                             const T &gamma ) const {
            std::void_type void_obj;
            return run<false, true>( var, info, void_obj, gamma );
        }

        /**
         * @see run( const variable_t& var, metainfo_t &info,
         *        typename std::conditional<feasibility_enabled, feasibility_t, std::void_type >::type &feasibility,
         *        typename std::conditional<gamma_enabled, T, std::void_type >::type level ) const
         */

        std::tuple<variable_t,T> operator()( const variable_t& var, metainfo_t &info ) const {
            std::void_type void_obj;
            return run<false,false>( var, info, void_obj, void_obj );
        }









        /**
         * @brief compute gradient of objectiv function
         * @tparam feasibility_enabled enable/disable feasibility checking
         * @tparam gamma_enabled enable/disable set init hyperparameter gamma
         * @param var variable
         * @param info metainfo
         * @param line feasibility check
         * @param level hyperparameter
         * @return gradients
         */

        template<bool feasibility_enabled = false, bool gamma_enabled = false>
        inline std::tuple<variable_t,T> run( const variable_t& var, metainfo_t &info,
                  typename std::conditional<feasibility_enabled, feasibility_t, std::void_type >::type &feasibility,
                  typename std::conditional<gamma_enabled, T, std::void_type >::type level ) const {
            variable_t gradient; T objective = 0;

            std::for_range<0,L::value>([&]<auto I>(){
                std::get<I>(gradient).weight = 0;
                std::get<I>(gradient).bias = 0;
            });

            T gamma = 1.0;
            if constexpr ( gamma_enabled )
                gamma = level;

            // compute gradient
            std::invoke( &self_back_t::run, *this, var, info, gradient, objective);
            auto L = std::invoke( &self_barrier_t::compute, *this, var, gradient, gamma );


            // prepare feasibility check
            if constexpr ( feasibility_enabled )
                    feasibility.init( std::move(L),  std::invoke( &self_barrier_t::tparam, *this ) );

            return std::make_tuple( std::move(gradient) ,
                                    std::move(objective) );
        }




    };

}

#endif // __LIPNET_NETWORK_PROBLEM_LOG_BARRIER_WOT_HPP__
