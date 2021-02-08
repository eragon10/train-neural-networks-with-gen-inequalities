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

#ifndef __LIPNET_NETWORK_LIPTRAIN_ENFORCING_HPP__
#define __LIPNET_NETWORK_LIPTRAIN_ENFORCING_HPP__

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



    /// helper function for usage of mosek api
    template<size_t ...N, typename ARGS>
    auto flatten_admm_( const ARGS &arg ) {
        auto res = monty::new_array_ptr<double,1>( sum_mul_pair<N...>() );
        size_t index = 0;


        std::for_range<0, sizeof... (N)-1>([&]<auto I>(){
            auto& mat = std::get<I>(arg).weight;
            for( size_t i=0UL; i< mat.rows(); ++i )
                for( size_t j=0UL; j< mat.columns(); ++j )
               {
                    (*res)[index] = mat(i,j); index++; }

        });

        return fusion::Matrix::dense( 1, sum_mul_pair<N...>(), res )->transpose();
    }

    /// helper function for usage of mosek api
    template<size_t N>
    auto flatten_admm_( const std::array< fusion::Variable::t, N> &arg ) {
        auto stack = monty::new_array_ptr< fusion::Expression::t, 1>( N );

        std::for_range<0, N>([&]<auto I>(){
            auto mat = std::get<I>(arg);
            (*stack)[I] = fusion::Expr::flatten( mat );
        });

        return fusion::Expr::vstack( stack );
    }


    /// helper function for usage of mosek api
    template<typename T, size_t N>
    auto to_diag_matrix_admm_( const blaze::StaticVector<T, N, blaze::columnVector> &m) {
        auto values = monty::new_array_ptr<double,1>( N );
        auto rindex = monty::new_array_ptr<int,1>( N );
        auto cindex = monty::new_array_ptr<int,1>( N );

        for( size_t i=0UL; i< N; ++i ) {
            (*values)[i] = m.at(i); (*rindex)[i] = i; (*cindex)[i] = i; }

        return fusion::Matrix::sparse( N, N, rindex, cindex, values );
    }


    /// helper function for usage of mosek api
    template<size_t ...N>
    auto generate_b_block_admm_() {
        constexpr size_t n = sum_from_to<1, sizeof... (N)-1, N...>();

        auto values = monty::new_array_ptr<double,1>( n );
        auto rindex = monty::new_array_ptr<int,1>( n );
        auto cindex = monty::new_array_ptr<int,1>( n );

        for( size_t i=0UL; i< n; ++i ) {
            (*values)[i] = 1.0; (*rindex)[i] = i; (*cindex)[i] = at<0,N...>()+i; }

        auto hh = (int) sum_from_to<0,sizeof... (N)-1,N...>();
        return fusion::Matrix::sparse((int)n, (int) sum_from_to<0,sizeof... (N),N...>(),
                    rindex, cindex, values);
    }

    /// helper function for usage of mosek api
    auto zero_mat_admm_( size_t r, size_t c) {
        return fusion::Expr::repeat( fusion::Expr::zeros(r), c, 1 );
    }

    
    /// helper function for usage of mosek api
    template<size_t N>
    inline auto block_diag_admm_( const std::array< fusion::Expression::t, N > &list ) {
        static_assert ( N > 0, "have to provide same data");

        if constexpr (N == 1) return std::get<0>(list);

        auto stack = monty::new_array_ptr< fusion::Expression::t, 1>( N );

        size_t w = 0;
        std::for_range<0, N >([&]<auto I>(){  w += std::get<I>(list)->getDim(1);});

        size_t r = 0; size_t c = 0;
        std::for_range<0, N>([&]<auto I>(){
            auto mat = std::get<I>(list);

            if constexpr (I <= 0) {
                (*stack)[I] = fusion::Expr::hstack(mat,
                         zero_mat( mat->getDim(0),  w-mat->getDim(1) ) ); }

            if constexpr (I >= N-1)
                (*stack)[I] = fusion::Expr::hstack(
                      zero_mat(  mat->getDim(0),  w-mat->getDim(1) ) , mat  );

            if constexpr (I > 0 && I < N-1)
                (*stack)[I] = fusion::Expr::hstack(
                      zero_mat(  mat->getDim(0),  r ) , mat,
                                zero_mat(  mat->getDim(0),  w-mat->getDim(1)-r )  );


            r += mat->getDim(0); c += mat->getDim(1);
        });

        return fusion::Expr::vstack( stack );

    }


    /**
     * @brief network_libtrain_enforcing_; Implementaion of the second subproblem
     *          of the admm method
     * 
     *          @f[  \arg \min_{\tilde{W},\eta} \quad \mathrm{tr}(Y(W- \tilde{W})) + \frac{v}{2} \eta  \quad \mathrm{s.t.}
     *                      \quad \chi(\Psi^2,\tilde{W}) \succeq 0 \quad \left[  \eta \;\;\; \mathrm{fl}(W- \tilde{W}) \right] \succeq_\mathcal{Q} 0  @f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     * 
     */

    template<typename T, size_t ...N>
    struct network_libtrain_enforcing_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;


        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;

        typedef std::integral_constant<size_t, sum_from_to<1, L::value, N...>() > n;
        typedef typename network_t<T, identity_activation_t, N...>::layer_t variable_t;


        /**
         * @brief solve second admm subproblem
         * @param lipschitz lipschitz constant
         * @param mu admm hyperparameter; augmented lagrange multipliers
         * @param Rvar refernce weights \f$ W \f$
         * @param SDT hyperparameter \f$T\f$ of matrix \f$\chi(\Psi^2,W)\f$
         * @param dual dual variable
         */

        static variable_t train(  const T lipschitz, const T mu, const variable_t &Rvar,
                                  const vector_t<n::value> &SDT, const variable_t &dual ) {

            // create mosek model
            auto M = new fusion::Model("liptrain");

            // create mosek variables to optimize
            std::array<fusion::Variable::t, L::value> Wvar;
            std::for_range<0,L::value>([&]<auto I>(){
                std::get<I>(Wvar) = M->variable( std::format("W%i", I),
                      monty::new_array_ptr<int,1>({ at<I+1,N...>() ,  at<I,N...>() }) );
            });

            
            // generate mosek parameters from input arguments
            auto Tparam = to_diag_matrix_admm_( SDT );

            auto weights = flatten_admm_<N...>( Rvar );
            auto weights_bar = flatten_admm_( Wvar );
            auto y = flatten_admm_<N...>( dual );

            std::array<fusion::Expression::t, L::value-1> wtmp;
            std::transform(std::begin(Wvar), std::prev(std::end(Wvar)), std::begin(wtmp),
                           [](auto &var) { return var->asExpr(); });

            
            // generate matricies A and B to generate chi at the end
            auto A = fusion::Expr::hstack( block_diag_admm_( wtmp ),
                    zero_mat_admm_( n::value, sum_from_to<L::value-1, L::value+1, N...>() ) );
            auto B = generate_b_block_admm_<N...>();



            // generate all parts for chi
            auto term1 = fusion::Expr::mul(  B->transpose(),  fusion::Expr::mul( Tparam, A) );
            auto term2 = fusion::Expr::transpose( term1 );
            auto term3 = fusion::Expr::mul( -2.0, fusion::Expr::mul( B->transpose(), fusion::Expr::mul(
                                                          fusion::Expr::constTerm(Tparam), B) ));

            
            // generate Q
            auto qupper = fusion::Expr::mul(  -std::pow( lipschitz, 2), fusion::Expr::constTerm( fusion::Matrix::eye( at<0,N...>() )) );

            auto qlower = fusion::Expr::vstack( fusion::Expr::hstack( zero_mat_admm_( at<L::value-1,N...>(), at<L::value-1,N...>() ),
                                                                      fusion::Expr::transpose(std::get<L::value-1>(Wvar)) ),
                                                fusion::Expr::hstack( std::get<L::value-1>(Wvar),
                                                                      fusion::Expr::neg( fusion::Expr::constTerm(
                                                                                             fusion::Matrix::eye( at<L::value,N...>() )) ) ) );
            // generate Q
            constexpr size_t zerolen = n::value - at<L::value-1, N...>();
            fusion::Expression::t term4;
            if constexpr ( zerolen <= 0) {
                std::array<fusion::Expression::t,2> ptr = {qupper, qlower};
                term4 = block_diag_admm_( ptr );
            }
            if constexpr ( zerolen > 0) {
                std::array<fusion::Expression::t,3> ptr = {qupper,
                        zero_mat_admm_(zerolen, zerolen), qlower};
                term4 = block_diag_admm_( ptr );
            }


            
            // generate matrix ch
            auto P = fusion::Expr::add( monty::new_array_ptr<fusion::Expression::t,1>( {term1, term2, term3, term4}) );


            // set psd contraint to mosel model
            M->constraint( fusion::Expr::neg(P), fusion::Domain::inPSDCone() );


            // create additional  variable fro quadratic cone
            auto Nvar = M->variable("Norm", fusion::Domain::greaterThan(0.));
            auto diff = fusion::Expr::flatten( fusion::Expr::mul( 1.0, fusion::Expr::add(
                                        fusion::Expr::sub( weights, weights_bar ), y)) );

            // set quadratic cone contraint
            M->constraint( fusion::Expr::vstack( 0.5, Nvar, diff ), fusion::Domain::inRotatedQCone() );


            // set objective function
            M->objective( fusion::ObjectiveSense::Minimize, fusion::Expr::add( fusion::Expr::sum( diff ) ,
                                                                               fusion::Expr::mul( mu , Nvar)) );
            
            // solve mosek problem
            M->solve();
            

            // extract weights from solution
            variable_t result = Rvar;
            std::for_range<0,L::value>([&]<auto I>(){
                auto& mat = std::get<I>(result).weight;
                for( size_t i=0UL; i< mat.rows(); ++i )
                    for( size_t j=0UL; j< mat.columns(); ++j ) {
                        mat(i,j) = (* std::get<I>(Wvar)->level() )[i*mat.columns()+j]; }
            });


            return std::move(result);


        }



    };

}

#endif // __LIPNET_NETWORK_LIPTRAIN_ENFORCING_HPP__
