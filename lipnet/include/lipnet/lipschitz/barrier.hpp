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

#ifndef __LIPNET_LIPSCHITZ_BARRIER_HPP__
#define __LIPNET_LIPSCHITZ_BARRIER_HPP__

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
#include <ratio>

#include <cereal/cereal.hpp>

#include <lipnet/traits.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/tuple.hpp>
#include <lipnet/variable.hpp>

#include <lipnet/network/network.hpp>

#include <lipnet/lipschitz/topology.hpp>

namespace lipnet {



    template<typename T, size_t ...N>
    struct liptrainweights_t{
        typename network_topology<T,N...>::type W;
        typename parameter_tparam<T,N...>::type t;

        template <class Archive>
        void save( Archive & ar) const
        {
            std::array<size_t, sizeof... (N)> topo = { N... };
            ar( cereal::make_nvp("topology", topo) );

            std::for_range<0, sizeof ...(N)-1>([&]<auto I>(){
                ar( cereal::make_nvp( std::format("l-%i", I) , std::get<I>(W) ) );
            });

            std::for_range<0, sizeof ...(N)-2>([&]<auto I>(){
                ar( cereal::make_nvp( std::format("t-%i", I) , std::get<I>(t) ) );
            });
        }


        template <class Archive>
        void load( Archive & ar )
        {
        }

    };



    template<typename T, size_t ...N>
    inline liptrainweights_t<T,N...>& operator-=(liptrainweights_t<T,N...> &a,
                                                 const liptrainweights_t<T,N...> &b) {
        a.W -= b.W; a.t -= b.t; return a; }


    template<typename T, size_t ...N>
    inline liptrainweights_t<T,N...>& operator+=(liptrainweights_t<T,N...> &a,
                                                 const liptrainweights_t<T,N...> &b) {
          a.W += b.W; a.t += b.t; return a; }



    template<typename T, size_t ...N>
    inline auto operator*(const T&a, const liptrainweights_t<T,N...> &b) {
        return liptrainweights_t<T,N...>{ a * b.W, a * b.t };
    }

    template<typename T, size_t ...N>
    inline auto operator+(const T&a, const liptrainweights_t<T,N...> &b) {
        return liptrainweights_t<T,N...>{ a + b.W, a + b.t };
    }



    template<typename T, size_t ...N>
    inline auto operator*(const liptrainweights_t<T,N...> &a, const liptrainweights_t<T,N...> &b) {
        return liptrainweights_t<T,N...>{ a.W * b.W, a.t * b.t };
    }

    template<typename T, size_t ...N>
    inline auto operator/(const liptrainweights_t<T,N...>  &a, const liptrainweights_t<T,N...>  &b) {
        return liptrainweights_t<T,N...>{ a.W / b.W, a.t / b.t };
    }


    template<typename T, size_t ...N>
    inline auto operator-(const liptrainweights_t<T,N...> &a, const liptrainweights_t<T,N...> &b) {
        return liptrainweights_t<T,N...>{ a.W - b.W, a.t - b.t };
    }

    template<typename T, size_t ...N>
    inline auto operator+(const liptrainweights_t<T,N...> &a, const liptrainweights_t<T,N...> &b) {
        return liptrainweights_t<T,N...>{ a.W + b.W, a.t + b.t };
    }



    template<typename T, size_t ...N>
    struct generator_t<liptrainweights_t<T,N...>> {
        static inline liptrainweights_t<T,N...> make(T val, T uni) {
            return liptrainweights_t<T,N...>{
                generator_t<decltype (liptrainweights_t<T,N...>::W)>::make( val ),
                generator_t<decltype (liptrainweights_t<T,N...>::t)>::make( uni )
            };
        }
    };


    template<typename T, size_t ...N>
    struct norm_t<T, liptrainweights_t<T,N...>> {
        typedef decltype (liptrainweights_t<T,N...>::W) arg1_t;
        typedef decltype (liptrainweights_t<T,N...>::t) arg2_t;

        static inline T norm( const liptrainweights_t<T,N...> &m ) {
            return norm_t<T,arg1_t>::norm(m.W) + norm_t<T,arg2_t>::norm(m.t);
        }
    };


    template<typename T, size_t ...N>
    struct function_t<liptrainweights_t<T,N...>> {
        typedef decltype (liptrainweights_t<T,N...>::W) arg1_t;
        typedef decltype (liptrainweights_t<T,N...>::t) arg2_t;

        static inline auto square( const liptrainweights_t<T,N...> &m ) {
            return liptrainweights_t<T,N...>{
                function_t<arg1_t>::square( m.W ),
                function_t<arg2_t>::square( m.t )
            };
        }

        static inline auto sqrt( const liptrainweights_t<T,N...> &m ) {
            return liptrainweights_t<T,N...>{
                function_t<arg1_t>::sqrt( m.W ),
                function_t<arg2_t>::sqrt( m.t )
            };
        }
    };


    template<typename T, size_t ...N>
    struct prod_t<T, liptrainweights_t<T,N...>, liptrainweights_t<T,N...>> {
        typedef decltype (liptrainweights_t<T,N...>::W) arg1_t;
        typedef decltype (liptrainweights_t<T,N...>::t) arg2_t;

        static inline T inner( const liptrainweights_t<T,N...> &m1, const liptrainweights_t<T,N...> &m2 ) {
            return prod_t<T,arg1_t,arg1_t>::inner(m1.W,m2.W) + prod_t<T,arg2_t,arg2_t>::inner( m1.t, m2.t);
        }
    };







    /**
     * @brief implementation of the log barrier function
     *          
     *          @f[ \mu(W,T) = - \log \det ( \chi(\Psi^2,W,T) )  @f]
     * 
     * @tparam T numerical value type
     * @tparam N network topology
     */

    template<typename T, size_t ...N>
    struct barrierfunction_t {

        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef blaze::IdentityMatrix<T> eye;

        typedef typename cholesky_topology<T,N...>::type cholesky_t;
        typedef typename inverse_topology<T,N...>::type inverse_t;
        typedef typename network_topology<T,N...>::type weights_t;
        typedef typename parameter_tparam<T,N...>::type tparam_t ;

        typedef liptrainweights_t<T,N...> variable_t;

        typedef std::integral_constant<size_t, sizeof... (N)-2> LN;
        typedef std::integral_constant<size_t, sizeof... (N)-1> L;

        /// lipschitz constant
        T lipschitz;

        
        /**
         * @brief barrierfunction_t; default constructor
         * @param lipschitz lipschitz constant
         */ 
        
        explicit barrierfunction_t( const T lipschitz = 70.0 )
            : lipschitz{ lipschitz } {}


        /**
         * @brief compute gradients 
         * @param var current position
         * @param gradient reuturn value gradient
         * @param gamma hyperparameter of barrier function
         */
        
        auto compute( const variable_t& var,  variable_t& gradient, const T& gamma ) const {

            cholesky_t L = chol( lipschitz, var );
            inverse_t P = inv( L );

            std::for_range<0,L::value>([&]<auto I>(){
                auto& grad = std::get<I>( gradient.W ).weight;
                auto& submat = std::get<I>( P.K );

                if constexpr ( I < L::value-1 )
                    grad += 2*gamma*blaze::expand<at<I,N...>()>( std::get<I>( var.t ) ) % submat;

                if constexpr ( I == L::value-1 )
                    grad += 2*gamma*submat;
            });

            std::for_range<0,L::value-1>([&]<auto I>(){
                 auto& submat = std::get<I>( P.K );
                 auto& subdiag = std::get<I+1>( P.P );
                 auto& w = std::get<I>( var.W ).weight;

                 auto& grad = std::get<I>( gradient.t );

                 grad += 2*( blaze::diagonal(submat*blaze::trans(w)) - blaze::diagonal(subdiag) );

            });

            return std::move( L );
        }

        
        /**
         * @brief execute cholesky decomposition
         * @tparam numeric_stability enable/disable numerical offset
         * @tparam kondition numerical offset
         * @param lipschitz lipschitz constant
         * @param var current position
         */

        template<bool numeric_stability = true, typename kondition = std::ratio<1,100>,
                 typename = typename std::enable_if<kondition::den != 0>::type>
        inline cholesky_t chol(const T lipschitz, const variable_t &var ) const {
            constexpr T ratio = ( (T) kondition::num )/( (T) kondition::den );

            cholesky_t value;

            if constexpr ( numeric_stability )
                std::get<0>( value.D ) = lipschitz;
            else std::get<0>( value.D ) = lipschitz;

            std::get<0>( value.L ) =  -blaze::trans(
                         blaze::expand<at<0,N...>()>( blaze::trans( std::get<0>( var.t ) )) %
                         blaze::trans( std::get<0>( var.W ).weight ) / lipschitz );

            std::for_range<1,LN::value>([&]<auto I>(){
                typedef matrix_t<at<I,N...>(),at<I,N...>()> smatrix_t;

                smatrix_t X; blaze::diagonal( X ) = 2*std::get<I-1>( var.t );
                X -= std::get<I-1>( value.L ) * blaze::trans(
                                 std::get<I-1>( value.L ) );

                if constexpr ( numeric_stability )
                    X += ratio*eye( at<I,N...>() );

                blaze::llh( X , std::get<I>( value.D ) );

                auto Z = blaze::trans( std::get<I>( var.W ).weight )
                           % blaze::expand<at<I,N...>()>( blaze::trans( std::get<I>( var.t ) ) );
                std::get<I>( value.L ) = - blaze::trans(
                      blaze::solve( std::get<I>( value.D ), Z ) );

            });

            typedef matrix_t<at<LN::value,N...>(), at<LN::value,N...>()> s1matrix_t;
            typedef matrix_t<at<LN::value+1,N...>(), at<LN::value+1,N...>()> s2matrix_t;

            s1matrix_t X1; blaze::diagonal( X1 ) = 2*std::get<LN::value-1>( var.t );
            X1 -= std::get<LN::value-1>( value.L ) * blaze::trans(
                             std::get<LN::value-1>( value.L ) );

            if constexpr ( numeric_stability )
                X1 += ratio*eye( at<LN::value,N...>() );

            blaze::llh( X1 , std::get<LN::value>( value.D ) );


            std::get<LN::value>( value.L ) = - blaze::trans(
                  blaze::solve( std::get<LN::value>( value.D ), blaze::trans(
                                    std::get<LN::value>( var.W  ).weight  ) ) );

            s2matrix_t X2 = eye( at<LN::value+1,N...>() );

            if constexpr ( numeric_stability )
                X2 += ratio*eye( at<LN::value+1,N...>() );

            X2 -= std::get<LN::value>( value.L ) * blaze::trans(
                             std::get<LN::value>( value.L ) );
            blaze::llh( X2 , std::get<LN::value+1>( value.D ) );

            return std::move( value );
        }


        /**
         * @brief compute inverse_t
         * @param val cholesky decomposition (e.g L)
         */
        
        inline inverse_t inv( const cholesky_t &val ) const {
            typedef matrix_t<at<LN::value+1,N...>(),
                    at<LN::value+1,N...>()> imat;

            inverse_t res;

            imat identy = eye( at<LN::value+1,N...>() );
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

                imat identy = eye( at<LN::value-I,N...>() );
                imat temp = blaze::solve( D, blaze::decldiag(identy) );
                P =  blaze::solve( blaze::trans( D ) , blaze::decllow( temp ) )
                                             - blaze::trans( tmp*K );
            });

            
            std::get<0>( res.K ) = - blaze::trans( std::get<1>(res.P) )
                    * std::get<0>(val.L)  / std::get<0>(val.D);

            std::get<0>( res.P ) = eye( at<0,N...>() ) / std::pow( std::get<0>(val.D), 2)
                    - blaze::trans( std::get<0>( res.K ) ) *  std::get<0>(val.L)  / std::get<0>(val.D);

            return std::move( res );
        }

    };

}

#endif // __LIPNET_LIPSCHITZ_BARRIER_HPP__
