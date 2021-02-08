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

#ifndef __LIPNET_LIPSCHITZ_BARRIER_WOT_HPP__
#define __LIPNET_LIPSCHITZ_BARRIER_WOT_HPP__

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

#include <lipnet/lipschitz/topology.hpp>

namespace lipnet {





    template<typename T, size_t ...N>
    struct barrierfunction_wot_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef blaze::IdentityMatrix<T> eye;

        typedef typename cholesky_topology<T,N...>::type  cholesky_t;
        typedef typename inverse_topology<T,N...>::type inverse_t;
        typedef typename network_topology<T,N...>::type variable_t;
        typedef typename parameter_tparam<T,N...>::type tparam_t ;

        typedef std::integral_constant<size_t, sizeof... (N)-2> LN;
        typedef std::integral_constant<size_t, sizeof... (N)-1> L;

        T lipschitz;
        tparam_t tparam;

        explicit barrierfunction_wot_t( tparam_t&& tmat, const T lipschitz = 70.0 )
            : tparam{ std::move(tmat) }, lipschitz{ lipschitz } {}


        auto compute( const variable_t& var,  variable_t& gradient, const T& gamma ) const {
            cholesky_t L = chol( lipschitz, var, tparam );
            inverse_t P = inv( L );

            std::for_range<0,L::value>([&]<auto I>(){
                auto& grad = std::get<I>( gradient ).weight;
                auto& submat = std::get<I>( P.K );

                if constexpr ( I < L::value-1 )
                    grad += 2*gamma*blaze::expand<at<I,N...>()>( std::get<I>( tparam ) ) % submat;

                if constexpr ( I == L::value-1 )
                    grad += 2*gamma*submat;
            });

            return std::move( L );

        }

       inline cholesky_t chol(const T lipschitz, const variable_t &weights,
                              const tparam_t &tparam ) const {

            cholesky_t value;

            std::get<0>( value.D ) = lipschitz;
            std::get<0>( value.L ) =  -blaze::trans(
                         blaze::expand<at<0,N...>()>( blaze::trans( std::get<0>( tparam ) )) %
                         blaze::trans( std::get<0>( weights ).weight ) / lipschitz );

            std::for_range<1,LN::value>([&]<auto I>(){
                typedef matrix_t<at<I,N...>(),at<I,N...>()> smatrix_t;

                smatrix_t X; blaze::diagonal( X ) = 2*std::get<I-1>(tparam);
                X -= std::get<I-1>( value.L ) * blaze::trans(
                                 std::get<I-1>( value.L ) );
                blaze::llh( X , std::get<I>( value.D ) );

                auto Z = blaze::trans( std::get<I>( weights ).weight )
                           % blaze::expand<at<I,N...>()>(  blaze::trans( std::get<I>(tparam) ) );
                std::get<I>( value.L ) = - blaze::trans(
                      blaze::solve( std::get<I>( value.D ), Z ) );

            });

            typedef matrix_t<at<LN::value,N...>(), at<LN::value,N...>()> s1matrix_t;
            typedef matrix_t<at<LN::value+1,N...>(), at<LN::value+1,N...>()> s2matrix_t;

            s1matrix_t X1; blaze::diagonal( X1 ) = 2*std::get<LN::value-1>(tparam);
            X1 -= std::get<LN::value-1>( value.L ) * blaze::trans(
                             std::get<LN::value-1>( value.L ) );
            blaze::llh( X1 , std::get<LN::value>( value.D ) );


            std::get<LN::value>( value.L ) = - blaze::trans(
                  blaze::solve( std::get<LN::value>( value.D ), blaze::trans(
                                    std::get<LN::value>( weights ).weight  ) ) );

            s2matrix_t X2 = blaze::IdentityMatrix<T>( at<LN::value+1,N...>() );
            X2 -= std::get<LN::value>( value.L ) * blaze::trans(
                             std::get<LN::value>( value.L ) );
            blaze::llh( X2 , std::get<LN::value+1>( value.D ) );

            return std::move( value );
        }


        inline inverse_t inv( const cholesky_t &val ) const {
            typedef matrix_t<at<LN::value+1,N...>(),
                    at<LN::value+1,N...>()> imat;

            inverse_t res;

            imat identy = blaze::IdentityMatrix<T>( at<LN::value+1,N...>() );
            imat temp =  blaze::solve( std::get<LN::value+1>( val.D ), blaze::decldiag(identy)  );
            std::get<LN::value+1>( res.P ) = blaze::solve( blaze::trans(
                        std::get<LN::value+1>( val.D ) ) , blaze::decllow( temp )  );

            std::for_range<0, LN::value>([&]<auto I>(){
                typedef matrix_t<at<LN::value-I,N...>(),
                        at<LN::value-I,N...>()> imat;

                auto& K = std::get<LN::value-I>( res.K );
                auto& Pp = std::get<LN::value-I+1>( res.P );
                auto& P = std::get<LN::value-I>( res.P );

                auto& L = std::get<LN::value-I>( val.L );
                auto& D = std::get<LN::value-I>( val.D );

                auto tmp =  blaze::solve( blaze::declupp( blaze::trans(D) ),
                                             blaze::trans(L));
                K = -blaze::trans( tmp*Pp );

                imat identy = blaze::IdentityMatrix<T>( at<LN::value-I,N...>() );
                imat temp = blaze::solve( D, blaze::decldiag(identy) );
                P =  blaze::solve( blaze::trans( D ) , blaze::decllow( temp ) )
                                             - blaze::trans( tmp*K );
            });

            //std::cout << "sadsad: " << std::get<0>(val.D) << "\n";

            std::get<0>( res.K ) = - blaze::trans( std::get<1>(res.P) )
                    * std::get<0>(val.L)  / std::get<0>(val.D);

            std::get<0>( res.P ) = eye( at<0,N...>() ) / std::pow( std::get<0>(val.D), 2)
                    - blaze::trans( std::get<0>( res.K ) ) *  std::get<0>(val.L)  / std::get<0>(val.D);

            return std::move( res );
        }

    };

}

#endif // __LIPNET_LIPSCHITZ_BARRIER_WOT_HPP__
