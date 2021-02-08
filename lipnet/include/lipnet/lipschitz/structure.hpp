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

#ifndef __LIPNET_STRUCTURE_LIP_HPP__
#define __LIPNET_STRUCTURE_LIP_HPP__

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


#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>

#include <lipnet/lipschitz/topology.hpp>





namespace lipnet {

    template<typename T, size_t ...N, typename variable_t
                = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_train_a ( const variable_t &var ) {
        typedef std::integral_constant<size_t, sizeof... (N)-1 > L;
        typedef std::integral_constant<size_t, (N + ...)
               - at<0,N...>() - at<sizeof... (N)-1,N...>() > TN;
        typedef std::integral_constant<size_t, (N + ...) > NN;

        auto A = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    TN::value, NN::value );

        std::for_range<0,L::value-1>([&]<auto I>(){
            auto &weight = std::get<I>(var).weight;

            blaze::submatrix< sum<I+1,N...>() - at<0,N...>() , sum<I,N...>(),
                 at<I+1,N...>(), at<I,N...>() >( A ) = weight;
        });

        return std::move(A);
    }


    template<typename T, size_t ...N>
    inline auto generate_lipschitz_train_b() {
        typedef std::integral_constant<size_t, (N + ...)
               - at<0,N...>() - at<sizeof... (N)-1,N...>() > TN;
        typedef std::integral_constant<size_t, (N + ...)> NN;

        auto B = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    TN::value, NN::value );

        blaze::band< at<0,N...>() >( B ) = blaze::uniform( TN::value, 1.0 );

        return std::move(B);
    }



    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_train_q( const variable_t &weights, const T rho ) {
        typedef std::integral_constant<size_t, sizeof... (N)-1 > L;
        typedef std::integral_constant<size_t, (N + ...) > NN;


        auto Q = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    NN::value, NN::value );

        /*
         * Insert the last weight
         */

        blaze::submatrix<NN::value-at<L::value,N...>()-at<L::value-1,N...>(),
                 NN::value-at<L::value,N...>(), at<L::value-1,N...>(),
                 at<L::value,N...>() >( Q )
           = blaze::trans(std::get<L::value-1>(weights).weight);

        blaze::submatrix<NN::value-at<L::value,N...>(),
                 NN::value-at<L::value,N...>()-at<L::value-1,N...>(),
                 at<L::value,N...>(), at<L::value-1,N...>() >( Q )
           = std::get<L::value-1>(weights).weight;


        /*
         * Set the lipschitz constant and the identity
         */

        blaze::submatrix<0,0,at<0,N...>(),at<0,N...>()>( Q )
                 = - rho * blaze::IdentityMatrix<T>( at<0,N...>() );
        blaze::submatrix<NN::value-at<L::value,N...>(),NN::value-at<L::value,N...>(),
                at<L::value,N...>(), at<L::value,N...>()>( Q )
                        = -1.0*blaze::IdentityMatrix<T>( at<L::value,N...>() );


        return std::move(Q);

    }





    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_train_q_direction( const variable_t &weights ) {
        typedef std::integral_constant<size_t, sizeof... (N)-1 > L;
        typedef std::integral_constant<size_t, (N + ...) > NN;

        auto Q = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    NN::value, NN::value );

        blaze::submatrix<NN::value-at<L::value,N...>()-at<L::value-1,N...>(),
                 NN::value-at<L::value,N...>(), at<L::value-1,N...>(),
                 at<L::value,N...>() >( Q )
           = blaze::trans(std::get<L::value-1>(weights).weight);

        blaze::submatrix<NN::value-at<L::value,N...>(),
                 NN::value-at<L::value,N...>()-at<L::value-1,N...>(),
                 at<L::value,N...>(), at<L::value-1,N...>() >( Q )
           = std::get<L::value-1>(weights).weight;

        return  std::move(Q);

    }


    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_train_t( const typename parameter_tparam<T,N...>::type &tparam ) {

        typedef std::integral_constant<size_t, (N + ...)
                - at<0,N...>() - at<sizeof... (N)-1,N...>() > NN;

        auto TT = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    NN::value, NN::value );

        std::for_range<0,sizeof... (N)-2>([&]<auto I>(){
            blaze::subvector<sum<I+1,N...>()-at<0,N...>(),at<I+1,N...>()>(
                    blaze::diagonal( TT )) = std::get<I>( tparam );
        });

        return  std::move(TT);

    }



    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_train_l( const typename cholesky_topology<T,N...>::type &lower ) {

        typedef std::integral_constant<size_t, (N + ...) > NN;

        auto L = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    NN::value, NN::value );

        blaze::submatrix<0,0, at<0,N...>(), at<0,N...>() >( L )
                = blaze::IdentityMatrix<T>( at<0,N...>() ) * std::get<0>( lower.D );
        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            blaze::submatrix<sum<I+1,N...>(),sum<I,N...>(),
              at<I+1,N...>(), at<I,N...>() >( L ) = std::get<I>( lower.L );

            blaze::submatrix<sum<I+1,N...>(),sum<I+1,N...>(),
              at<I+1,N...>(), at<I+1,N...>() >( L ) = std::get<I+1>( lower.D );
        });

        return  std::move(L);

    }




    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline void extract_lipschitz_train_p( const blaze::StaticMatrix<double, (N + ...), (N + ...),
                                           blaze::columnMajor> &p,
                                           const blaze::StaticVector<double, (N + ...) - at<0,N...>()
                                                    - at<sizeof... (N)-1,N...>() > &Tmat, variable_t &weights ) {

         typedef std::integral_constant<size_t, sizeof... (N)-1 > L;
         typedef std::integral_constant<size_t, (N + ...) > NN;


        std::for_range<0,L::value>([&]<auto I>(){
            auto &W = std::get<I>( weights ).weight;

            auto Psub = blaze::submatrix<sum<I+1,N...>(), sum<I,N...>(),
                                      at<I+1,N...>() , at<I,N...>() >( p );


            if constexpr ( I < L::value-1 ) {
                auto Tsub = 1.0 / blaze::subvector<sum<I+1,N...>() - at<0,N...>(),
                                      at<I+1,N...>() >(Tmat);
                W = -1.0 * blaze::expand<at<I,N...>()>( Tsub ) % Psub;
            }

            if constexpr ( I == L::value )
                W = -1.0 * Psub;
        });

    }































    template<typename T, size_t ...N, typename variable_t
                = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_calc_a ( const variable_t &var ) {
        typedef std::integral_constant<size_t, (N + ...)
               - at<0,N...>() - at<sizeof... (N)-1,N...>() > TN;
        typedef std::integral_constant<size_t, (N + ...)
               - at<sizeof... (N)-1,N...>() > QN;


        auto A = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    TN::value, QN::value );

        std::for_range<0,sizeof... (N)-2>([&]<auto I>(){
            auto &weight = std::get<I>(var).weight;

            blaze::submatrix< sum<I+1,N...>() - at<0,N...>() , sum<I,N...>(),
                 at<I+1,N...>(), at<I,N...>() >( A ) = weight;
        });

        return std::move(A);
    }


    template<typename T, size_t ...N>
    inline auto generate_lipschitz_calc_b() {
        typedef std::integral_constant<size_t, (N + ...)
               - at<0,N...>() - at<sizeof... (N)-1,N...>() > TN;
        typedef std::integral_constant<size_t, (N + ...)
               - at<sizeof... (N)-1,N...>() > QN;

        auto B = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    TN::value, QN::value );

        blaze::band<at<0,N...>()>( B ) = blaze::uniform( TN::value, 1.0 );

        return std::move(B);
    }


    template<typename T, size_t ...N, typename variable_t
             = typename network_topology<T, N...>::type >
    inline auto generate_lipschitz_calc_q( const variable_t &weights, const T rho ) {
        typedef std::integral_constant<size_t,sizeof... (N)-1> L;

        typedef std::integral_constant<size_t, (N + ...)
               - at<0,N...>() - at<sizeof... (N)-1,N...>() > TN;
        typedef std::integral_constant<size_t, (N + ...)
                - at<sizeof... (N)-1,N...>()  > QN;

        auto Q = blaze::CompressedMatrix<T, blaze::columnMajor>(
                    QN::value, QN::value );


        auto& W = std::get<L::value-1>(weights).weight;
        blaze::submatrix<QN::value-at<L::value-1,N...>(),
                 QN::value-at<L::value-1,N...>(), at<L::value-1,N...>(),
                 at<L::value-1,N...>() >( Q ) = blaze::trans(W) * W;

        blaze::submatrix<0,0, at<0,N...>(), at<0,N...>()>( Q )
                 = - rho * blaze::IdentityMatrix<T>( at<0,N...>() );

        return std::move(Q);
    }




}

#endif // __LIPNET_STRUCTURE_LIP_HPP__
