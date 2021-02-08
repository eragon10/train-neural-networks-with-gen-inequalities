 
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

#ifndef __LIPNET_LIP_HELPER_HPP__
#define __LIPNET_LIP_HELPER_HPP__

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
#include <lipnet/variable.hpp>

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>


#include <fusion.h>
#include <mosek.h>

using namespace mosek;



namespace lipnet {


    template<size_t START, size_t... Ints, size_t... Seq>
    constexpr size_t sum_from_to_(std::integral_constant<size_t,START>,
                          std::integer_sequence<size_t, Ints...>,
                          std::integer_sequence<size_t, Seq...> ) {

            constexpr size_t arr[] = {Ints...};
            return (arr[START+Seq] + ... );
    }

    template<size_t F, size_t T, size_t... Ints>
    constexpr size_t sum_from_to() {
        return sum_from_to_(  std::integral_constant<size_t, F>{},
                      std::integer_sequence<size_t, Ints...>{},
                      std::make_integer_sequence<size_t,T-F>{});
    }




    template<size_t... Ints, size_t... Seq>
    constexpr size_t sum_mul_pair_(std::integer_sequence<size_t, Ints...>,
                          std::integer_sequence<size_t, Seq...> ) {

            constexpr size_t arr[] = {Ints...};
            return ( (arr[1+Seq]*arr[Seq]) + ... );
    }

    template<size_t... Ints>
    constexpr size_t sum_mul_pair() {
        return sum_mul_pair_( std::integer_sequence<size_t, Ints...>{},
                      std::make_integer_sequence<size_t, sizeof... (Ints)-1>{});
    }


    template<size_t ...INTS, typename ...ARGS>
    auto extract_weights_( std::integer_sequence<size_t,INTS...>,
                           const std::tuple<ARGS...> &args ) {
        return std::make_tuple( std::get<INTS>(args).weight ... );
    }

    template<typename ...ARGS>
    auto extract_weights( const std::tuple<ARGS...> &args ) {
        return extract_weights_( std::make_integer_sequence<size_t, sizeof... (ARGS)>{},
                                 args);
    }


    template<size_t ...N>
    auto block_diag_b_const( std::integer_sequence<size_t,N...> ) {
        constexpr size_t n = sum_from_to<1, sizeof... (N)-1, N...>();

        auto values = monty::new_array_ptr<double,1>( n );
        auto rindex = monty::new_array_ptr<int,1>( n );
        auto cindex = monty::new_array_ptr<int,1>( n );

         for( size_t i=0UL; i< n; ++i ) {
            (*values)[i] = 1.0; (*rindex)[i] = i; (*cindex)[i] = at<0,N...>()+i; }

         auto hh = (int) sum_from_to<0,sizeof... (N)-1,N...>();

        return fusion::Matrix::sparse((int)n, (int) sum_from_to<0,sizeof... (N)-1,N...>(),
                    rindex, cindex, values);
    }


    template<size_t ...N, typename ...ARGS>
    auto block_diag_a_const( std::integer_sequence<size_t,N...>, const std::tuple<ARGS...> &arg ) {
        constexpr size_t n = sum_from_to<1, sizeof... (N)-1, N...>();

        auto values = monty::new_array_ptr<double,1>( sum_mul_pair<N...>() );
        auto rindex = monty::new_array_ptr<int,1>( sum_mul_pair<N...>() );
        auto cindex = monty::new_array_ptr<int,1>( sum_mul_pair<N...>() );


        size_t index = 0;
        std::for_range<0, sizeof... (N)-2>([&]<auto I>(){
            auto w = std::get<I>(arg);

            for( size_t i=0UL; i< w.rows(); ++i )
                for( size_t j=0UL; j< w.columns(); ++j ) {
                    (*values)[index] = std::get<I>(arg)(i,j);
                    (*rindex)[index] = sum<I+1,N...>() - at<0,N...>() + i;
                    (*cindex)[index] = sum<I,N...>() + j;
                    index++;
                }
        });

        return fusion::Matrix::sparse((int)n, (int) sum<sizeof... (N)-1,N...>(),
                    rindex, cindex, values);
    }


    template<size_t ...N, typename ARG>
    auto block_diag_q( std::integer_sequence<size_t,N...>, const ARG &arg , fusion::Variable::t &L) {
        constexpr size_t n = sum_from_to<0, sizeof... (N)-1, N...>();
        constexpr size_t q = at<0,N...>();

        auto valuesq = monty::new_array_ptr<double,1>( q );
        auto rindexq = monty::new_array_ptr<int,1>( q );
        auto cindexq = monty::new_array_ptr<int,1>( q );

        for(int i = 0; i < q ; ++i){
            (*valuesq)[i] = -1; (*rindexq)[i] = i; (*cindexq)[i] = i;}

        auto varQ = fusion::Matrix::sparse(n, n, rindexq, cindexq, valuesq);


        constexpr size_t nn = at<sizeof... (N)-2, N...>()*at<sizeof... (N)-2, N...>();
        auto values = monty::new_array_ptr<double,1>( nn );
        auto rindex = monty::new_array_ptr<int,1>( nn );
        auto cindex = monty::new_array_ptr<int,1>( nn );


        size_t index = 0;
        for( size_t i=0UL; i< arg.rows(); ++i )
            for( size_t j=0UL; j< arg.columns(); ++j ) {
                (*values)[index] = arg(i,j);
                (*rindex)[index] = n - arg.rows() + i ;
                (*cindex)[index] = n - arg.columns() + j ;
                index++;
            }

        auto constQ = fusion::Matrix::sparse(n, n, rindex, cindex, values);

        auto tmp = fusion::Expr::mulElm(  fusion::Expr::repeat( fusion::Expr::repeat(L, n, 1)  , n, 0) , varQ );

        return fusion::Expr::add( tmp , constQ );
    }

    auto zeros( size_t r, size_t c) {
        return fusion::Expr::repeat( fusion::Expr::zeros(r), c, 1 );
    }
    
        auto zero_mat( size_t r, size_t c) {
        return fusion::Expr::repeat( fusion::Expr::zeros(r), c, 1 );
    }

}

#endif // __LIPNET_LIP_HELPER_HPP__
