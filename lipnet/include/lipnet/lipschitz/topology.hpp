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

#ifndef __LIPNET_LIPSCHITZ_TOPOLOGY_HPP__
#define __LIPNET_LIPSCHITZ_TOPOLOGY_HPP__

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


    template<typename T, size_t N, size_t ...NS>
    struct parameter_tparam_impl {
      typedef typename parameter_tparam_impl<T, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::StaticVector<T,N,blaze::columnVector>>, next>::type type;
    };

    template<typename T, size_t N, size_t R>
    struct parameter_tparam_impl<T, N, R>{
        typedef std::tuple<blaze::StaticVector<T,N,blaze::columnVector>> type; };


    template<typename T, size_t N, size_t ...NARGS>
    struct parameter_tparam {
        typedef typename parameter_tparam_impl<T, NARGS...>::type type;
    };








             

    /**
     * @see cholesky_diagentry
     */
    
    template<typename T, size_t N, size_t ...NS>
    struct cholesky_diagentry_impl {
      typedef typename cholesky_diagentry_impl<T, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::LowerMatrix<blaze::StaticMatrix<T,N,N>>>, next>::type type;
    };

    /**
     * @see cholesky_diagentry
     */
    
    template<typename T, size_t N>
    struct cholesky_diagentry_impl<T, N>{
        typedef std::tuple<blaze::LowerMatrix<blaze::StaticMatrix<T,N,N>>> type; };


    /**
     * @brief data holder for cholesky decomposition;
     *          only diagonal elements
     * 
     * @tparam T numerical value type
     * @tparam N matrix dimension
     * @tparam NARGS passthrough dimensions 
     */
        
    template<typename T, size_t N, size_t ...NARGS>
    struct cholesky_diagentry {
        typedef typename cholesky_diagentry_impl<T, NARGS...>::type next;
        typedef typename join_tuples<std::tuple<T>, next>::type type;
    };




    /**
     * @see cholesky_subentry
     */
    
    template<typename T, size_t NI, size_t NO, size_t ...NS>
    struct cholesky_subentry_impl {
      typedef typename cholesky_subentry_impl<T, NO, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::StaticMatrix<T,NO,NI>>, next>::type type;
    };

    /**
     * @see cholesky_subentry
     */
    
    template<typename T, size_t NI, size_t NO>
    struct cholesky_subentry_impl<T, NI, NO>{
        typedef std::tuple<blaze::StaticMatrix<T,NO,NI>> type; };

    /**
     * @brief data holder for cholesky decomposition;
     *          only subdiagonal elements
     * 
     * @tparam NI input dimension / column dimension
     * @tparam NO output dimension / row dimension
     * @tparam RE compile test dimension (same as NARGS)
     * @tparam NARGS passthrough dimensions
     */
    
    template<typename T, size_t NI, size_t NO, size_t RE, size_t ...NARGS>
    struct cholesky_subentry {
        typedef typename cholesky_subentry_impl<T, NI, NO, RE, NARGS...>::type type;
    };

    
    /**
     * @brief combined data holder of diagonal and subdiagonal
     *          elements; cholesky_subentry and cholesky_diagentry
     * @tparam T numerical value type
     * @tparam N dimensions
     * @see cholesky_diagentry
     * @see cholesky_subentry
     */
    
    template<typename T, size_t ...N>
    struct cholesky_topology {
        typedef struct {
            typename cholesky_diagentry<T, N...>::type D;
            typename cholesky_subentry<T,N...>::type L;
        } type;
    };







    /**
     * @see inverse_diagentry
     */

    template<typename T, size_t N, size_t ...NS>
    struct inverse_diagentry_impl {
      typedef typename inverse_diagentry_impl<T, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::SymmetricMatrix<blaze::StaticMatrix<T,N,N>>>, next>::type type;
    };
    
    /**
     * @see inverse_diagentry
     */

    template<typename T, size_t N>
    struct inverse_diagentry_impl<T, N>{
        typedef std::tuple<blaze::SymmetricMatrix<blaze::StaticMatrix<T,N,N>>> type; };


    /**
     * @brief data holder for inverse computation;
     *          onöy diagonal elements
     * @tparam T numerical value type
     * @tparam N matrix dimension
     * @tparam NARGS passthrough dimensions
     */
        
    template<typename T, size_t N, size_t ...NARGS>
    struct inverse_diagentry {
        typedef typename inverse_diagentry_impl<T, N, NARGS...>::type type;
    };

    
    
    /**
     * @see inverse_subentry
     */
    
    template<typename T, size_t NI, size_t NO, size_t ...NS>
    struct inverse_subentry_impl {
      typedef typename inverse_subentry_impl<T, NO, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::StaticMatrix<T,NO,NI>>, next>::type type;
    };
    
    /**
     * @see inverse_subentry
     */

    template<typename T, size_t NI, size_t NO>
    struct inverse_subentry_impl<T, NI, NO>{
        typedef std::tuple<blaze::StaticMatrix<T,NO,NI>> type; };

    /**
     * @brief data holder for inverse computation;
     *          only subdiagonal elements
     * @tparam T numerical value type
     * @tparam NI input dimension / column dimension
     * @tparam NO output dimension / row dimension
     * @tparam RE compile time test dimension (like NARGS)
     * @tparam NARGS passthrough dimensions
     */
    
    template<typename T, size_t NI, size_t NO, size_t RE, size_t ...NARGS>
    struct inverse_subentry {
        typedef typename inverse_subentry_impl<T, NI, NO, RE, NARGS...>::type type;
    };


    /**
     * @brief combined data holder of diagonal and subdiagonal
     *          elements; inverse_subentry and inverse_diagentry
     * @tparam T numerical value type
     * @tparam N dimensions
     * @see inverse_subentry
     * @see inverse_diagentry
     */
    
    template<typename T, size_t ...N>
    struct inverse_topology {
        typedef struct {
            typename inverse_diagentry<T, N...>::type P;
            typename inverse_subentry<T,N...>::type K;
        } type;
    };


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    // helper function to debug
    template<typename T, size_t ...N>
    inline void print_inverse_topology( std::ostream &stream, const typename inverse_topology<T,N...>::type &val ) {
        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            stream << "DIAG(" << I << ")\n" << std::get<I>( val.P ) << "\n";
        });

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "SUB(" << I << ")\n" << std::get<I>( val.K ) << "\n";
        });

    }
    
    // helper function to debug
    template<typename T, size_t ...N>
    inline void print_cholesky_topology( std::ostream &stream, const typename cholesky_topology<T,N...>::type &val ) {
        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            stream << "DIAG(" << I << ")\n" << std::get<I>( val.D ) << "\n";
        });

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "SUB(" << I << ")\n" << std::get<I>( val.L ) << "\n";
        });

    }


}

#endif // __LIPNET_LIPSCHITZ_TOPOLOGY_HPP__
