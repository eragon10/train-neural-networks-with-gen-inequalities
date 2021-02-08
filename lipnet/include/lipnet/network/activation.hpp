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

#ifndef __LIPNET_ACTIVATION_HPP__
#define __LIPNET_ACTIVATION_HPP__

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




namespace lipnet {

    /**
     * @brief The atype_t enum; all possible activation functions
     */

    enum atype_t{ SIGMOID, TANH, NONE };

    /**
     * @brief The activation_t struct; implementation of the activation functions
     * @tparam T numerical value type
     * @tparam TYPE choose the activation type
     */

    template<typename T, atype_t TYPE>
    struct activation_t {

        template<typename TT, size_t O, size_t I>
        using matrix_t =  blaze::StaticMatrix<TT,O,I, blaze::columnMajor>;
        template<typename TT, size_t N>
        using vector_t = blaze::StaticVector<TT,N,blaze::columnVector>;

        /**
         * @brief evaluate activation function
         * @tparam N input dimension
         * @tparam BATCH batch size
         * @param val input vector
         * @return output vector
         */

        template<size_t N, size_t BATCH = 1>
        static inline auto forward( const auto& val) {
            typedef typename std::conditional<BATCH <= 0,
                    vector_t<T,N>, matrix_t<T, N, BATCH>>::type TT;
            typedef typename std::conditional<BATCH <= 0,
                    vector_t<bool,N>, matrix_t<bool, N, BATCH>>::type MT;

            /// \f$ \sigma(x) = \frac{1}{ 1 + \exp{-x} } \f$
            if constexpr ( TYPE == atype_t::SIGMOID)
                return 1 / ( 1 + blaze::exp( -val ) );

            /// \f$ \sigma(x) = \tanh(x) \f$
            if constexpr ( TYPE == atype_t::TANH)
                return blaze::tanh(val);

            /// \f$ \sigma(x) = x \f$
            if constexpr ( TYPE == atype_t::NONE)
                return val;
        }

        /**
         * @brief derivative of activation function
         * @tparam N input dimension
         * @tparam BATCH batch size
         * @param val input vector
         * @return output vector
         */

        template<size_t N, size_t BATCH = 1>
        static inline auto derivative( const auto& val ) {
            typedef typename std::conditional<BATCH <= 0,
                    vector_t<T,N>, matrix_t<T, N, BATCH>>::type TT;
            typedef typename std::conditional<BATCH <= 0,
                    vector_t<bool,N>, matrix_t<bool, N, BATCH>>::type MT;

            if constexpr ( TYPE == atype_t::SIGMOID) {
                auto sig = 1 / ( 1 + blaze::exp( -val) );
                auto sigp =  1 - sig;
                return sig * sigp;
            }

            if constexpr ( TYPE == atype_t::TANH)
                return  1 - blaze::pow(blaze::tanh(val), 2);

            if constexpr ( TYPE == atype_t::NONE)
                return TT(1);

        }

    };





    template<typename TT>
    using tanh_activation_t = activation_t<TT, atype_t::TANH>;

    template<typename TT>
    using sigmoid_activation_t = activation_t<TT, atype_t::SIGMOID>;

    template<typename TT>
    using identity_activation_t = activation_t<TT, atype_t::NONE>;

}

#endif // __LIPNET_ACTIVATION_HPP__
