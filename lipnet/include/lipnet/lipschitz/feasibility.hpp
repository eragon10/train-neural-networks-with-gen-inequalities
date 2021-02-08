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

#ifndef __LIPNET_LIPSCHITZ_FEASIBILITY_HPP__
#define __LIPNET_LIPSCHITZ_FEASIBILITY_HPP__

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
#include <lipnet/optimizer.hpp>

#include <lipnet/network/network.hpp>

#include <lipnet/lipschitz/topology.hpp>
#include <lipnet/lipschitz/structure.hpp>
#include <lipnet/lipschitz/barrier.hpp>


namespace lipnet {



    /**
     * @brief feasibilitycheck_wot_t; Implementation of the feasibility check 
     *          for eigenvalue problem (not quadratic)
     * 
     *          @f[ \det( P - \alpha D ) = 0 \quad \quad P - \alpha D = \chi(\Psi^2,W - \alpha \Delta W)@f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     */

    template<typename T, size_t ...N>
    struct feasibilitycheck_wot_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef blaze::IdentityMatrix<T> eye;

        typedef typename cholesky_topology<T,N...>::type cholesky_t;
        typedef typename inverse_topology<T,N...>::type inverse_t;
        typedef typename network_topology<T,N...>::type variable_t;
        typedef typename parameter_tparam<T,N...>::type tparam_t ;

        typedef std::integral_constant<size_t, (N + ...) > NN;
        typedef std::integral_constant<size_t, sizeof... (N)-1> L;


        /**
         * @brief solve eigenvalue problem
         * @param tparam hyperparamater T of matrix chi
         * @param var cholesky decomposition of matrix P
         * @param gradient update direction e.g. matrix D
         */
        
        T compute( const tparam_t& tparam, const cholesky_t& var, const variable_t& gradient ) const {

            auto B = generate_lipschitz_train_b<T,N...>();
            auto A = generate_lipschitz_train_a<T,N...>( gradient );
            auto Q = generate_lipschitz_train_q_direction<T,N...>( gradient );
            auto Z = generate_lipschitz_train_t<T,N...>( tparam );
            matrix_t<NN::value,NN::value> L = generate_lipschitz_train_l<T,N...>( var );


            matrix_t<NN::value,NN::value> D = blaze::trans(B)*Z*A + blaze::trans(A)*Z*B + Q;
            matrix_t<NN::value,NN::value> V = blaze::trans( blaze::solve( L, D ) );
            matrix_t<NN::value,NN::value> R = - blaze::solve( L, V ); // -

            auto vals = blaze::map( blaze::real( blaze::eigen( R ) ), [](const T& val){
                if(val < 0) return 0.01;
                return val;
            });
            T eigenvalue = blaze::max( vals );
            return 1.0 / (std::abs(eigenvalue) + 0.001);

        }

    };


    /**
     * @brief feasibilitycheck_t; Implementation of the feasibility check 
     *          for generalized eigenvalue problem (e.g. quadratic eigenvalue problem)
     * 
     *              @f[ \det(P - \alpha D + \alpha^2 M) = 0 \quad \quad P - \alpha D + \alpha^2 M
     *                       = \chi(\Psi^2, W - \alpha \Delta W, T - \alpha Delta T) @f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     */

    template<typename T, size_t ...N>
    struct feasibilitycheck_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef blaze::IdentityMatrix<T> eye;

        typedef typename cholesky_topology<T,N...>::type cholesky_t;
        typedef typename inverse_topology<T,N...>::type inverse_t;
        typedef typename network_topology<T,N...>::type weight_t;
        typedef typename parameter_tparam<T,N...>::type tparam_t ;
        typedef liptrainweights_t<T,N...> variable_t;


        typedef std::integral_constant<size_t, (N + ...) > NN;
        typedef std::integral_constant<size_t, sizeof... (N)-1> L;


        /**
         * @brief solve generalized eigenvalue problem
         * @tparam kondition init matrix N with that value
         * @param pos current position 
         * @param gradient update direction
         * @param rho squared lipschitz constant
         */
        
        template<typename kondition = std::ratio<2,1>,
                 typename = typename std::enable_if<kondition::den != 0>::type>
        T compute( const variable_t& pos, const variable_t& gradient, const T rho ) const {
            constexpr T ratio = ( (T) kondition::num )/( (T) kondition::den );

            auto B = generate_lipschitz_train_b<T,N...>();
            auto Ag = generate_lipschitz_train_a<T,N...>( gradient.W );
            auto Qg = generate_lipschitz_train_q_direction<T,N...>( gradient.W );
            auto Zg = generate_lipschitz_train_t<T,N...>( gradient.t );

            auto Ap = generate_lipschitz_train_a<T,N...>( pos.W );
            auto Qp = generate_lipschitz_train_q<T,N...>( pos.W, rho );
            auto Zp = generate_lipschitz_train_t<T,N...>( pos.t );


            matrix_t<2*NN::value,2*NN::value> A, C;

            blaze::submatrix<0,NN::value,NN::value,NN::value>(A) =
                    ratio*blaze::IdentityMatrix<T>( NN::value );
            blaze::submatrix<NN::value,NN::value,NN::value,NN::value>(A) =
                    blaze::trans(B)*Zp*Ag + blaze::trans(Ag)*Zp*B + Qg
                  + blaze::trans(B)*Zg*Ap + blaze::trans(Ap)*Zg*B
                  - 2*blaze::trans(B)*Zg*B;
            blaze::submatrix<NN::value,0,NN::value,NN::value>(A) =
                    blaze::trans(B)*Zp*Ap + blaze::trans(Ap)*Zp*B + Qp
                  - 2*blaze::trans(B)*Zp*B;


            blaze::submatrix<0,0,NN::value,NN::value>(C) =
                    ratio*blaze::IdentityMatrix<T>( NN::value );
            blaze::submatrix<NN::value,NN::value,NN::value,NN::value>(C) =
                    - blaze::trans(B)*Zg*Ag - blaze::trans(Ag)*Zg*B;


            blaze::StaticVector<std::complex<T>,2*NN::value,blaze::columnVector> alpha;
            vector_t<2*NN::value>  beta;
            blaze::gges( A , C, alpha, beta );



            auto rr = blaze::map( alpha, beta,  []( std::complex<T> a, T b ){
                    if( std::abs(a.imag()) < 1e-6 && std::abs(b) > 1e-6 ) {
                        auto val = a.real() / b;
                        if( val < 0) return val;
                        else return -1.0;
                    }
                    else return -1.0;
            });

            return std::abs(blaze::max( rr ));
        }

    };

}

#endif // __LIPNET_LIPSCHITZ_FEASIBILITY_HPP__
