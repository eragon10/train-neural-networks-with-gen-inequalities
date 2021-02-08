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

#ifndef __LIPNET_LIPSCHITZ_TRIVIAL_HPP__
#define __LIPNET_LIPSCHITZ_TRIVIAL_HPP__

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

#include <lipnet/network/network.hpp>





namespace lipnet {

    /**
     * @brief compute trivial lipschitz constant
     * @tparam T numerical value type
     * @tparam N network topology
     * @cite gouk2020regularisation
     */

    template<typename T, size_t ...N>
    struct calculate_lipschitz_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef typename network_topology<T, N...>::type variable_t;


        static T trivial_lipschitz( const variable_t &var ) {

            T lipschitz = 1.0; blaze::DynamicVector<T,blaze::columnVector> sigma;

            std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
                auto sigma = blaze::svd( std::get<I>( var ).weight );
                lipschitz *= blaze::max( sigma );
            });

            return lipschitz;

        }
    };


}

#endif // __LIPNET_LIPSCHITZ_TRIVIAL_HPP__
