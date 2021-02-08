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

#ifndef __LIPNET_NETWORK_LIPCALC_HPP__
#define __LIPNET_NETWORK_LIPCALC_HPP__

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


#include <lipnet/extern/lip_helper.hpp>

#include <fusion.h>
#include <mosek.h>

using namespace mosek;







namespace lipnet {

    
    /**
     * @breif calculate lipschitz constant of neural network via
     *          conic program (SDP)
     * 
     *      @f[ \arg \min_{\Psi,T} \quad \Psi^2 \quad \mathrm{s.t} \chi(\Psi^2,W) \succeq 0 @f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     * @cite fazlyab2019efficient
     */
    

    template<typename T, size_t ...N>
    struct network_libcalc_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;


        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;

        typedef typename network_topology<T, N...>::type variable_t;



        /**
         * @brief solve sdp; via mosek; via interior point method
         * @param var network weights
         * @cite fazlyab2019efficient
         */
        
        static std::tuple<T, vector_t<sum_from_to<1, L::value, N...>()>> solve( const variable_t& var ) {
            constexpr size_t n = sum_from_to<1, L::value, N...>();

            // create mosek model
            fusion::Model::t M = new fusion::Model("sdo1");
            auto _M = monty::finally([&]() { M->dispose(); });

            // create mosek variables to optimize
            fusion::Variable::t Lvar  = M->variable("L^2", 1, fusion::Domain::greaterThan(0.));
            fusion::Variable::t Tvar  = M->variable("T", n, fusion::Domain::greaterThan(0.));

            // generate network representation in matrices A and B
            auto A = block_diag_a_const( std::integer_sequence<size_t, N...>{}, extract_weights(var) );
            auto B = block_diag_b_const( std::integer_sequence<size_t, N...>{} );


            auto Tcon = fusion::Expr::repeat(Tvar, n+at<0,N...>(), 1);

            // gernate 
            auto term1 = fusion::Expr::mul( B->transpose(), fusion::Expr::mulElm( Tcon, A) );
            auto term2 = fusion::Expr::mul( A->transpose(), fusion::Expr::mulElm( Tcon, B) );
            auto term3 = fusion::Expr::mul( -2.0, fusion::Expr::mul( B->transpose(), fusion::Expr::mulElm( Tcon, B) ));

            auto res = blaze::trans(  std::get<L::value-1>(var).weight ) *  std::get<L::value-1>(var).weight;
            auto term4 = block_diag_q( std::integer_sequence<size_t, N...>{} , res , Lvar );

            // generate chi matrix
            auto P = fusion::Expr::add( monty::new_array_ptr<fusion::Expression::t,1>({term1, term2, term3, term4}) );


            // set objective function and contraint
            M->objective( fusion::ObjectiveSense::Minimize,  Lvar );
            M->constraint( fusion::Expr::neg(P), fusion::Domain::inPSDCone() );

            // solve mosek problem
            M->solve();

            // extract T param and psi (aka lipschitz constant)
            vector_t<sum_from_to<1, L::value, N...>()> TT;
            for(int i = 0; i < n; i++)
                TT.at(i) = (*Tvar->level())[i];

            // free mosek model
            M->dispose();

            return std::make_tuple( std::sqrt( (*(Lvar->level()))[0] ), std::move(TT)  );

        }
        

    };

}

#endif // __LIPNET_NETWORK_LIPCALC_HPP__
