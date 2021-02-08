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

#ifndef __LIPNET_NETWORK_TOPOLOGY_HPP__
#define __LIPNET_NETWORK_TOPOLOGY_HPP__

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
#include <lipnet/tuple.hpp>
#include <lipnet/variable.hpp>

#include <lipnet/network/layer.hpp>




namespace lipnet {

    /**
     * @brief newtork layer holder and creator struct; helper struct to create compile time
     *          layers in stack memory -> performance
     */

    template<typename T, size_t NI, size_t NO, size_t ...NS>
    struct network_topology_impl {
      typedef typename network_topology_impl<T, NO, NS...>::type next;
      typedef typename join_tuples<std::tuple<layer_t<T, NI, NO>>, next>::type type;
    };

    /**
     * @see network_topology_impl
     */
    template<typename T, size_t NI, size_t NO>
    struct network_topology_impl<T, NI, NO>{ // layer_t<T,NI,NO,TYPE>
        typedef std::tuple<layer_t<T,NI,NO>> type; };


    /**
     * @see network_topology_impl
     */
    template<typename T, size_t NI, size_t NO, size_t ...NARGS>
    struct network_topology {
        typedef typename network_topology_impl<T, NI, NO, NARGS...>::type type;
    };




    /**
     * @brief generator_t implementation for std::tuple
     * @see lipnet::generator_t
     */
    template<typename ...ARGS>
    struct generator_t<std::tuple<ARGS...>> {
        template<typename T>
        static inline std::tuple<ARGS...> make(T val) {
            return std::make_tuple( generator_t<ARGS>::make(val) ... );
        }
    };









    /// helper struct for data
    template<typename T, size_t N, size_t ...NS>
    struct generate_data {

      template<size_t NN>
      using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

      typedef typename generate_data<T, NS...>::type next;
      typedef typename join_tuples<std::tuple<vector_t<N>>, next>::type type;
    };

    /// helper struct for data
    template<typename T, size_t N>
    struct generate_data<T, N>{
        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        typedef std::tuple<vector_t<N>> type; };

    /// helper struct for data
    template<typename T, size_t N, size_t ...NS>
    struct generate_data_remove_first {
        typedef typename generate_data<T, NS..., 0>::type type;
    };








    /// helper struct for data
    template<typename T, size_t B, size_t N, size_t ...NS>
    struct generate_batch_data {

      template<size_t NN1, size_t NN2>
      using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

      typedef typename generate_batch_data<T, B, NS...>::type next;
      typedef typename join_tuples<std::tuple<matrix_t<N, B>>, next>::type type;
    };

    /// helper struct for data
    template<typename T, size_t B, size_t N>
    struct generate_batch_data<T, B, N>{

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef std::tuple<matrix_t<N, B>> type; };

    /// helper struct for data
    template<typename T, size_t B, size_t N, size_t ...NS>
    struct generate_batch_data_remove_first {
        typedef typename generate_batch_data<T, B, NS...>::type type;
    };




}

#endif // __LIPNET_NETWORK_TOPOLOGY_HPP__
