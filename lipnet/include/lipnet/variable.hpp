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

#ifndef __LIPNET_VARIABLES_HPP__
#define __LIPNET_VARIABLES_HPP__

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

#include <blaze/Blaze.h>




namespace lipnet {


    /**
     * @brief The norm_t struct. Just a interface for all possible types.
     *        Compute norm of argument  @f[ ||V||_2  @f].
     * @tparam T numerical value type
     * @tparam V tensor type of argument
     */

    template<typename T, typename V>
    struct norm_t {};


    /**
     * @brief The function_t struct. Just a interface for all possible types.
     *        Apply function to tensor elementwise.
     * @tparam V tensor type of argument
     */

    template<typename V>
    struct function_t {};

    /**
     * @brief The prod_t struct. Just a interface for all possible types.
     *        Compute inner/outer/... products @f[ V_1 V_2^\top; \;\; V_1^\top V_2; \;\; \cdots  @f].
     * @tparam T numerical value type
     * @tparam V1 tensor type of first argument
     * @tparam V2 tensot type of second argument
     */

    template<typename T, typename V1, typename V2>
    struct prod_t {
        /*static inline T inner( const V1 &v1, const V2 &v2 ) {
            return blaze::sum( v1 % v2 );
        }

        static inline auto outer( const V1 &v1, const V2 &v2 ) {
            return blaze::outer( v1, v2 ) ;
        }*/
    };


    /**
     * @brief The equation_system_t struct. Just a interface for all possible types.
     *        Solve a system of equations @f[ A x = b @f].
     * @tparam V1 tensor type of first argument
     * @tparam V2 tensot type of second argument
     */

    template<typename V1, typename V2>
    struct equation_system_t {};



    /**
     * @brief The generator_t struct. Just a interface for all possible types.
     *        Instanciate tensor of type V.
     * @tparam V tensor type to create
     */

    template<typename V>
    struct generator_t {};













    template<typename T, typename V>
    struct helper_norm_t {
        inline constexpr static bool value = std::is_invocable_r<T,
                decltype(&norm_t<T,V>::norm), const V&>::value;
    };


    template<typename V>
    struct helper_function_t {
        inline constexpr static bool v1 = std::is_invocable_r<V,
                decltype(&function_t<V>::square), const V&>::value;

        inline constexpr static bool v2 = std::is_invocable_r<V,
                decltype(&function_t<V>::sqrt), const V&>::value;

        inline constexpr static bool value = v1 && v2;
    };

    template<typename T, typename V1, typename V2>
    struct helper_inner_t {
        inline constexpr static bool value = std::is_invocable_r<T,
                decltype(&prod_t<T,V1,V2>::inner), const V1&, const V2&>::value;
    };




}

#endif // __LIPNET_VARIABLES_HPP__
