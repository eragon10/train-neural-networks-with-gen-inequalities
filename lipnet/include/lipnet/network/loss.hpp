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

#ifndef __LIPNET_LOSS_HPP__
#define __LIPNET_LOSS_HPP__

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





namespace lipnet {

    /**
     * @brief The squared_error_t struct; implementation of the squarred error objective function
     * @tparam T numerical value type
     * @tparam TYPE choose the activation type
     */

    template<typename T>
    struct squared_error_t
    {
        template<typename TT, size_t O, size_t I>
        using matrix_t =  blaze::StaticMatrix<TT,O,I, blaze::columnMajor>;
        template<typename TT, size_t N>
        using vector_t = blaze::StaticVector<TT,N,blaze::columnVector>;


        /**
         * @brief The evaluate function; compute loss
         *       @f[ \mathcal{L}(x,y) = (x-y)^\top (x-y)  @f]
         * @tparam N input dimension type
         * @tparam BATCH batch size
         * @param target real value
         * @param estimated value
         * @return loss
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<!(BATCH <= 0),int>::type = 0>
        T evaluate(const matrix_t<T,N, BATCH> &target, const matrix_t<T,N, BATCH> &data) const {
            matrix_t<T,N, BATCH> diff = data-target;
            return blaze::inner(diff,diff);
        }

        /**
         * @see evaluate(const matrix_t<T,N, BATCH> &target, const matrix_t<T,N, BATCH> &data)
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<BATCH <= 0,int>::type = 0>
        T evaluate(const  vector_t<T,N> &target, const  vector_t<T,N> &data) const {
            vector_t<T,N> diff = data-target;
            return blaze::trans(diff) * diff;
        }

        /**
         * @brief The gradient function; compute gradient of loss function
         *        @f[ \nabla_x \mathcal{L}(x,y) = 2(x-y)^\top @f]
         * @tparam N input dimension type
         * @tparam BATCH batch size
         * @param target real value
         * @param estimated value
         * @return gradient
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<!(BATCH <= 0),int>::type = 0>
        auto gradient(const matrix_t<T,N, BATCH> &target, const matrix_t<T,N, BATCH> &data) const {
            return 2*(data-target);
        }

        /**
         * @see gradient(const matrix_t<T,N, BATCH> &target, const matrix_t<T,N, BATCH> &data)
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<BATCH <= 0,int>::type = 0>
        auto gradient(const vector_t<T,N> &target, const vector_t<T,N> &data) const {
            return 2*(data-target);
        }

    };


    /**
     * @brief The cross_entropy_t struct; implementation of the cross entropy objective function
     *        @f[ \mathcal{L}(x,y) = \frac{ \sum [x == y] \exp{-x} }{ \sum \exp{-x} } @f]
     * @tparam T numerical value type
     * @tparam TYPE choose the activation type
     */

    template<typename T>
    struct cross_entropy_t
    {
        template<typename TT, size_t O, size_t I>
        using matrix_t =  blaze::StaticMatrix<TT,O,I, blaze::columnMajor>;
        template<typename TT, size_t N>
        using vector_t = blaze::StaticVector<TT,N,blaze::columnVector>;


        /**
         * @brief The evaluate function; compute loss
         * @tparam N input dimension type
         * @tparam BATCH batch size
         * @param target real value
         * @param estimated value
         * @return loss
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<!(BATCH <= 0),int>::type = 0>
        T evaluate(const matrix_t<T, N, BATCH> &target, const matrix_t<T, N, BATCH> &data) const {
            auto soft = blaze::softmax<blaze::columnwise>( data );
            matrix_t<T, N, BATCH> tmp = target % soft;
            return - blaze::sum( blaze::log(  blaze::reduce<blaze::columnwise>(tmp, blaze::Add() ) ));
        }

        /**
         * @see evaluate(const matrix_t<T, N, BATCH> &target, const matrix_t<T, N, BATCH> &data)
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<BATCH <= 0,int>::type = 0>
        T evaluate(const vector_t<T, N> &target, const vector_t<T, N> &data) const {
            auto soft = blaze::softmax( data );
            return - std::log( blaze::inner(target, soft) + 1e-8);
        }

        /**
         * @brief The gradient function; compute gradient of loss function
         * @tparam N input dimension type
         * @tparam BATCH batch size
         * @param target real value
         * @param estimated value
         * @return gradient
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<!(BATCH <= 0),int>::type = 0>
        auto gradient(const matrix_t<T, N, BATCH> &target, const matrix_t<T, N, BATCH> &data) const {
            auto soft = blaze::softmax<blaze::columnwise>( data );
            matrix_t<T, N, BATCH> o = soft - target;
            return o;
        }

        /**
         * @see gradient(const matrix_t<T, N, BATCH> &target, const matrix_t<T, N, BATCH> &data)
         */

        template<size_t N, size_t BATCH = 0, typename std::enable_if<BATCH <= 0,int>::type = 0>
        auto gradient(const vector_t<T, N> &target, const vector_t<T, N> &data) const {
            auto soft = blaze::softmax( data );
            vector_t<T, N> o = soft - target;
            return o;
        }

        
    };

}

#endif // __LIPNET_LOSS_HPP__
