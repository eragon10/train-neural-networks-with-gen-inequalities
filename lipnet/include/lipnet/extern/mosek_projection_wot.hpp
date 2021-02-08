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

#ifndef __MOSEK_PROJECTION_WOT_HPP__
#define __MOSEK_PROJECTION_WOT_HPP__

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
     * @brief The mosek_projection_wot_t struct. Compute the projection of the reference
     *          weights. It is conic program and will be solved with mosek (interior point method)
     * 
     *              @f[ \arg \min_{W,\eta} \quad \eta \quad \mathrm{s.t} \chi(\Psi^2,W) \succeq 0 \quad 
     *                      \left[ \eta \;\;\; \mathrm{fl}(W - W_\mathrm{ref} )  \right] \succeq_\mathcal{Q} 0 @f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     * 
     */

    template<typename T, size_t ...N>
    struct mosek_projection_wot_t {

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
         * @brief map input argument to mosek parameter type
         */
        
        template<size_t R, size_t C>
        static fusion::Matrix::t map( const matrix_t<R,C> &mat ) {
            auto values = monty::new_array_ptr<T,2>(
                    monty::shape( mat.rows(), mat.columns() ) );
            for( size_t i=0UL; i< mat.rows(); ++i )
                for( size_t j=0UL; j< mat.columns(); ++j ){
                            (*values)(i,j) = mat(i,j);  }
            return fusion::Matrix::dense( values );
        }



        /**
         * 
         * @brief Compute projetion of weights into feasible set.
         * @param lipschitz lipschitz integral_constant
         * @param ref reference weights; computed during gradient descent step
         * @param tinitval hyperparameter T of chi matrix
         */

        static variable_t projection(  const T lipschitz, variable_t &&ref, const T &tinitval ) {

            // creat emosek model
            auto M = new fusion::Model("ProjectionLipschitz");

            
            // create mosek variables
            std::array<fusion::Variable::t, L::value> Wvar;
            fusion::Variable::t Norm = M->variable("norm",
                       fusion::Domain::greaterThan(0.));

            std::array<fusion::Matrix::t, L::value> Wref;
            std::array<fusion::Matrix::t, L::value-1> Tparam;


            // generate mosek parameters from input argument
            std::for_range<0,L::value>([&]<auto I>(){
                std::get<I>(Wvar) = M->variable( std::format("var-%i", I),
                      monty::new_array_ptr<int,1>({ at<I+1,N...>() ,  at<I,N...>() }) );

                std::get<I>(Wref) = map<at<I+1,N...>(),at<I,N...>()>( std::get<I>(ref).weight );

                if constexpr ( I+1 < L::value )
                      std::get<I>(Tparam) = fusion::SparseMatrix::diag( at<I+1,N...>(), tinitval );
            });


            // stacking all parts together to generate chi matrix
            auto vstack = monty::new_array_ptr< fusion::Expression::t, 1>( L::value+1 );
            std::for_range<0,L::value+1>([&]<auto I>(){
                constexpr size_t nnnn = ( (I > 1 && I < L::value-1) ? 5 : (I > 0
                                          && I < L::value) ? 4 : 3);
                auto hstack = monty::new_array_ptr< fusion::Expression::t, 1>(nnnn);

                size_t i = 0;

                if constexpr ( I > 1 )
                     (*hstack)[i++] = zero_mat( at<I,N...>() ,  sum<I-1,N...>() );

                if constexpr ( I > 0 && I < L::value )
                     (*hstack)[i++] = fusion::Expr::neg( fusion::Expr::mul(
                                std::get<I-1>(Tparam), std::get<I-1>(Wvar)) );
                if constexpr ( I == L::value )
                     (*hstack)[i++] = fusion::Expr::neg( std::get<I-1>(Wvar) );

                if constexpr ( I == 0 )
                     (*hstack)[i++] = fusion::Expr::mul( fusion::Expr::constTerm(
                                    fusion::Matrix::eye( at<0,N...>() )), std::pow(lipschitz,2.0) );

                if constexpr ( I > 0 && I < L::value)
                     (*hstack)[i++] = fusion::Expr::mul( fusion::Expr::constTerm(std::get<I-1>(Tparam)), 2.0 );
                if constexpr ( I == L::value)
                     (*hstack)[i++] = fusion::Expr::constTerm( fusion::Matrix::eye( at<L::value,N...>() ) );

                if constexpr ( I < L::value-1 )
                     (*hstack)[i++] = fusion::Expr::transpose(  fusion::Expr::neg( fusion::Expr::mul(
                                std::get<I>(Tparam), std::get<I>(Wvar)) ));
                if constexpr ( I == L::value-1 )
                     (*hstack)[i++] = fusion::Expr::transpose(  fusion::Expr::neg( std::get<I>(Wvar) ));


                if constexpr ( I < L::value-1 )
                     (*hstack)[i++] = zero_mat( at<I,N...>() , sum<L::value+1,N...>()-sum<I+2,N...>() );

                (*vstack)[I] = fusion::Expr::hstack( hstack );
            });


            // stacking all flatted parts together to generate matrix for quadratic cone
            auto hstack = monty::new_array_ptr<
                   fusion::Expression::t, 1>( L::value+1 ); (*hstack)[0] = Norm;
            std::for_range<0,L::value>([&]<auto I>(){
                 (*hstack)[I+1]  = fusion::Expr::transpose( fusion::Expr::sub( fusion::Expr::flatten( std::get<I>(Wvar) ),
                        fusion::Expr::flatten( fusion::Expr::constTerm(std::get<I>(Wref)) ) ));
            });


            // final stacking
            auto D = fusion::Expr::hstack( hstack );
            auto P = fusion::Expr::vstack( vstack );

            // set constraints to mosek problem
            M->constraint(  P , fusion::Domain::inPSDCone() );
            M->constraint(  D , fusion::Domain::inQCone()  );


            // set objective function
            M->objective( fusion::ObjectiveSense::Minimize, Norm );
            
            // solve problem
            M->solve();
            
            
            // extract weights from solution
            variable_t result = std::move(ref);
            std::for_range<0,L::value>([&]<auto I>(){
                auto& mat = std::get<I>(result).weight;
                for( size_t i=0UL; i< mat.rows(); ++i )
                    for( size_t j=0UL; j< mat.columns(); ++j ) {
                        mat(i,j) = (* std::get<I>(Wvar)->level() )[i*mat.columns()+j]; }
            });


            // free mosel model
            M->dispose();

            return std::move(result);

        }



    };

}

#endif // __MOSEK_PROJECTION_WOT_HPP__
