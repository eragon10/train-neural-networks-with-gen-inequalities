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

#ifndef __LIPNET_TUPLE_HPP__
#define __LIPNET_TUPLE_HPP__

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
     * @brief Helper struct to join two tuples. (std::tuple)
     */

    template<typename, typename>
    struct join_tuples {};

    /**
     * @brief Implementation of join_tuples struct to join two tuples. (std::tuple)
     * @see lipnet::join_tuples
     */

    template<typename... NEW, typename... NEXT>
    struct join_tuples<std::tuple<NEW...>, std::tuple<NEXT...>>{
        typedef std::tuple<NEW..., NEXT...> type; };













    template<typename ...ARGS1, typename ...ARGS2, size_t ...INTS>
    inline auto add_impl( const std::tuple<ARGS1...> &a,
                        const std::tuple<ARGS2...> &b,
                        std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (std::get<INTS>(a)+std::get<INTS>(b)) ...) ;
    }
    template<typename ...ARGS1, typename ...ARGS2>
    inline auto operator+( const std::tuple<ARGS1...> &a, const std::tuple<ARGS2...> &b ) {
        static_assert ( sizeof... (ARGS1) == sizeof... (ARGS2), "not same size" );
        return add_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS1)>() );
    }


    template<typename ...ARGS1, typename ...ARGS2, size_t ...INTS>
    inline auto sub_impl( const std::tuple<ARGS1...> &a,
                        const std::tuple<ARGS2...> &b,
                        std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (std::get<INTS>(a)-std::get<INTS>(b)) ...) ;
    }
    template<typename ...ARGS1, typename ...ARGS2>
    inline auto operator-( const std::tuple<ARGS1...> &a, const std::tuple<ARGS2...> &b ) {
        static_assert ( sizeof... (ARGS1) == sizeof... (ARGS2), "not same size" );
        return sub_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS1)>() );
    }


    template<typename ...ARGS1, typename ...ARGS2, size_t ...INTS>
    inline auto mult_impl( const std::tuple<ARGS1...> &a,
                        const std::tuple<ARGS2...> &b,
                        std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (std::get<INTS>(a)*std::get<INTS>(b)) ...) ;
    }
    template<typename ...ARGS1, typename ...ARGS2>
    inline auto operator*( const std::tuple<ARGS1...> &a, const std::tuple<ARGS2...> &b ) {
        static_assert ( sizeof... (ARGS1) == sizeof... (ARGS2), "not same size" );
        return mult_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS1)>() );
    }


    template<typename ...ARGS1, typename ...ARGS2, size_t ...INTS>
    inline auto div_impl( const std::tuple<ARGS1...> &a,
                        const std::tuple<ARGS2...> &b,
                        std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (std::get<INTS>(a) / std::get<INTS>(b)) ...) ;
    }
    template<typename ...ARGS1, typename ...ARGS2>
    inline auto operator/( const std::tuple<ARGS1...> &a, const std::tuple<ARGS2...> &b ) {
        static_assert ( sizeof... (ARGS1) == sizeof... (ARGS2), "not same size" );
        return div_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS1)>() );
    }








    template<typename T, typename ...ARGS2, size_t ...INTS>
    inline auto add_simple_impl( const T &a,
                           const std::tuple<ARGS2...> &b,
                           std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (a+std::get<INTS>(b)) ...) ;
    }
    template<typename T, typename ...ARGS2>
    inline auto operator+( const T &a, const std::tuple<ARGS2...> &b ) {
        return add_simple_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS2)>() );
    }


    template<typename T, typename ...ARGS2, size_t ...INTS>
    inline std::tuple<ARGS2...> mult_simple_impl( const T &a,
                           const std::tuple<ARGS2...> &b,
                           std::integer_sequence<size_t, INTS...>) {
        return std::make_tuple( (a*std::get<INTS>(b)) ...) ;
    }
    template<typename T, typename ...ARGS2>
    inline std::tuple<ARGS2...> operator*( const T &a, const std::tuple<ARGS2...> &b ) {
        return mult_simple_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS2)>() );
    }








    template<typename ...ARGS, size_t ...INTS>
    inline void add_assign_impl( std::tuple<ARGS...> &a,
                            const std::tuple<ARGS...> &b,
                           std::integer_sequence<size_t, INTS...>) {
         ( ((void) (std::get<INTS>(a) += std::get<INTS>(b))), ... );
    }
    template<typename ...ARGS>
    std::tuple<ARGS...>& operator+=(std::tuple<ARGS...> &a, const std::tuple<ARGS...> &b) {
        add_assign_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS)>() );
        return a;
    }

    template<typename ...ARGS, size_t ...INTS>
    inline void sub_assign_impl( std::tuple<ARGS...> &a,
                            const std::tuple<ARGS...> &b,
                           std::integer_sequence<size_t, INTS...>) {
         ( ((void) (std::get<INTS>(a) -= std::get<INTS>(b))), ... );
    }
    template<typename ...ARGS>
    std::tuple<ARGS...>& operator-=(std::tuple<ARGS...> &a, const std::tuple<ARGS...> &b) {
        sub_assign_impl( a, b, std::make_integer_sequence<size_t, sizeof... (ARGS)>() );
        return a;
    }














    template<typename T, typename ...ARGS>
    struct norm_t<T, std::tuple<ARGS...>> {
        template<size_t ...INTS>
        static inline T norm_impl( const std::tuple<ARGS...> &m,
                                 std::integer_sequence<size_t, INTS...>) {
            return ( norm_t<T,ARGS>::norm( std::get<INTS>(m) ) + ... ) ;
        }

        static inline T norm( const std::tuple<ARGS...> &m ) {
            return norm_impl( m, std::make_integer_sequence<size_t, sizeof... (ARGS)>() );
        }
    };


    template<typename ...ARGS>
    struct function_t<std::tuple<ARGS...>> {
        template<size_t ...INTS>
        static inline auto square_impl( const std::tuple<ARGS...> &m,
                                 std::integer_sequence<size_t, INTS...>) {
            return std::make_tuple( function_t<ARGS>::square( std::get<INTS>(m) ) ... ) ;
        }

        static inline auto square( const std::tuple<ARGS...> &m ) {
            return square_impl( m, std::make_integer_sequence<size_t, sizeof... (ARGS)>() );
        }

        template<size_t ...INTS>
        static inline auto sqrt_impl( const std::tuple<ARGS...> &m,
                                 std::integer_sequence<size_t, INTS...>) {
            return std::make_tuple( function_t<ARGS>::sqrt( std::get<INTS>(m) ) ... ) ;
        }

        static inline auto sqrt( const std::tuple<ARGS...> &m ) {
            return sqrt_impl( m, std::make_integer_sequence<size_t, sizeof... (ARGS)>() );
        }

    };


    template<typename T, typename ...ARGS1, typename ...ARGS2>
    struct prod_t<T, std::tuple<ARGS1...>, std::tuple<ARGS2...>> {

        template<size_t ...INTS>
        static inline T inner_impl( const std::tuple<ARGS1...> &m1,
                                    const std::tuple<ARGS2...> &m2,
                                 std::integer_sequence<size_t, INTS...>) {
            return ( prod_t<T,ARGS1,ARGS2>::inner( std::get<INTS>(m1), std::get<INTS>(m2) ) + ... ) ;
        }

        static inline T inner( const std::tuple<ARGS1...> &m1, const std::tuple<ARGS2...> &m2 ) {
            static_assert ( sizeof... (ARGS1) == sizeof... (ARGS2), "not same size" );
            return inner_impl( m1, m2 , std::make_integer_sequence<size_t,sizeof... (ARGS1)>());
        }
    };


}

#endif // __LIPNET_TUPLE_HPP__
