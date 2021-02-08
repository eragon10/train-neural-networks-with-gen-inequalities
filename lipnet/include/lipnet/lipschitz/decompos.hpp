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

#ifndef __LIPNET_DECOMPOSITION_LIP_HPP__
#define __LIPNET_DECOMPOSITION_LIP_HPP__

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


    template<size_t... Ints, size_t... Seq>
    constexpr size_t folt_square_(std::integer_sequence<size_t, Ints...>,
                          std::integer_sequence<size_t, Seq...> ) {
        if constexpr( sizeof... (Seq) <= 0) {
                return 0;
        } else {
            constexpr size_t arr[] = {Ints...};
            return ((arr[Seq]*arr[Seq]) + ... );
        }
    }

    template<size_t I, size_t... Ints>
    constexpr size_t foldsquare() {
        static_assert ( sizeof... (Ints) >= 1, "have to provide at least one numbers");
        return folt_square_(  std::integer_sequence<size_t, Ints...>{},
                      std::make_integer_sequence<size_t,I>{});
    }


    template<size_t... Ints, size_t... Seq>
    constexpr size_t folt_mult_(std::integer_sequence<size_t, Ints...>,
                          std::integer_sequence<size_t, Seq...> ) {
        if constexpr( sizeof... (Seq) <= 0) {
                return 0;
        } else {
            constexpr size_t arr[] = {Ints...};
            return ((arr[Seq+1]*arr[Seq]) + ... );
        }
    }

    template<size_t I, size_t... Ints>
    constexpr size_t foldmult() {
        static_assert ( sizeof... (Ints) >= 2, "have to provide at least two numbers");
        return folt_mult_(  std::integer_sequence<size_t, Ints...>{},
                      std::make_integer_sequence<size_t,I>{}); // sizeof... (Ints)-1
    }



    template<typename T, size_t NI, size_t NO, size_t ...NS>
    struct decompos_subentry_impl {
      typedef typename decompos_subentry_impl<T, NO, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::StaticMatrix<T,NO,NI>>, next>::type type;
    };

    template<typename T, size_t NI, size_t NO>
    struct decompos_subentry_impl<T, NI, NO>{
        typedef std::tuple<blaze::StaticMatrix<T,NO,NI>> type; };


    template<typename T, size_t NI, size_t NO, size_t RE, size_t ...NARGS>
    struct decompos_subentry {
        typedef typename decompos_subentry_impl<T, NI, NO, RE, NARGS...>::type type;
    };






    // blaze::LowerMatrix<blaze::StaticMatrix<T,N,N>>
    template<typename T, size_t N, size_t ...NS>
    struct decompos_diagentry_impl {
      typedef typename decompos_diagentry_impl<T, NS...>::type next;
      typedef typename join_tuples<std::tuple<blaze::StaticMatrix<T,N,N>>, next>::type type;
    }; // blaze::LowerMatrix<blaze::StaticMatrix<T,N,N>>

    template<typename T, size_t N>
    struct decompos_diagentry_impl<T, N>{
        typedef std::tuple<blaze::StaticMatrix<T,N,N>> type; };
    // blaze::LowerMatrix<blaze::StaticMatrix<T,N,N>>

    template<typename T, size_t N, size_t ...NARGS>
    struct decompos_diagentry {
        typedef typename decompos_diagentry_impl<T, NARGS...>::type type;
    };









    template<typename T, size_t ...N>
    struct parameter_decompo_t {
        typename decompos_subentry<T, N...>::type subdiagonals;
        typename decompos_diagentry<T, N...>::type diagonals;
    };

    template<typename T, size_t ...N>
    struct generator_t<parameter_decompo_t<T,N...>> {
        static inline auto make( const T &init ) {
            return parameter_decompo_t<T,N...> {
                generator_t<typename decompos_subentry<T, N...>::type>::make( init ),
                generator_t<typename decompos_diagentry<T, N...>::type>::make( init )
            };
        }
    };

    template<typename T, size_t ...N>
    inline parameter_decompo_t<T,N...> operator*( const T &a, const parameter_decompo_t<T,N...> &b ) {
        return parameter_decompo_t<T,N...>{
            a*b.subdiagonals, a*b.diagonals
        };
    }

    template<typename T, size_t ...N>
    inline parameter_decompo_t<T,N...> operator+( const parameter_decompo_t<T,N...> &a,
                                                  const parameter_decompo_t<T,N...> &b ) {
        return parameter_decompo_t<T,N...>{
            a.subdiagonals+b.subdiagonals, a.diagonals+b.diagonals
        };
    }

    /*template<typename T, size_t ...N>
    struct parameter_decompo_dual_t {
        typename decompos_diagentry<T, N...>::type diagonals;
    };*/



    template<typename T,  size_t ...N>
    inline blaze::StaticVector<T, foldsquare<sizeof... (N),N...>()
                + foldmult<sizeof... (N)-1,N...>() - at<0,N...>()*at<0,N...>() , blaze::columnVector>
        parameter_flatten( const parameter_decompo_t<T,N...> &var ) {

        blaze::StaticVector<T, foldsquare<sizeof... (N),N...>()
                + foldmult<sizeof... (N)-1,N...>() - at<0,N...>()*at<0,N...>() , blaze::columnVector> res;

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            blaze::subvector<foldsquare<I+1,N...>() - at<0,N...>()*at<0,N...>()  , at<I+1,N...>()*at<I+1,N...>() >(res)
                 = blaze::flatten<T,at<I+1,N...>(),at<I+1,N...>()>( std::get<I>( var.diagonals ) );
        });

        typedef std::integral_constant<size_t, foldsquare<sizeof... (N),N...>()
                - at<0,N...>()*at<0,N...>() > OFFSET;

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            blaze::subvector<OFFSET::value + foldmult<I,N...>() , at<I,N...>()*at<I+1,N...>() >(res)
                 = blaze::flatten<T,at<I+1,N...>(),at<I,N...>()>( std::get<I>( var.subdiagonals ) );
        });

        return std::move(res);
    }


    template<typename T,  size_t ...N>
    inline parameter_decompo_t<T,N...> parameter_expansion( const blaze::StaticVector<T, foldsquare<sizeof... (N),N...>()
              + foldmult<sizeof... (N)-1,N...>() - at<0,N...>()*at<0,N...>() , blaze::columnVector> &var ) {

        parameter_decompo_t<T,N...> res;

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            std::get<I>( res.diagonals ) = blaze::tomatrix<T,at<I+1,N...>(),at<I+1,N...>()>(
                   blaze::subvector<foldsquare<I+1,N...>() - at<0,N...>()*at<0,N...>()  , at<I+1,N...>()*at<I+1,N...>() >(var) );
        });

        typedef std::integral_constant<size_t, foldsquare<sizeof... (N),N...>()
                - at<0,N...>()*at<0,N...>() > OFFSET;

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            std::get<I>( res.subdiagonals ) = blaze::tomatrix<T,at<I+1,N...>(),at<I,N...>()>(
                   blaze::subvector<OFFSET::value + foldmult<I,N...>() , at<I,N...>()*at<I+1,N...>() >(var));
        });

        return std::move(res);
    }




    template<typename T, size_t ...N>
    inline void print( std::ostream &stream, const typename decompos_diagentry<T, N...>::type &val ) {
        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "DIAG(" << I << ")\n" << std::get<I>( val ) << "\n";
        });

    };

    template<typename T, size_t ...N>
    inline void print( std::ostream &stream, const typename decompos_subentry<T, N...>::type &val ) {
        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "SUB(" << I << ")\n" << std::get<I>( val ) << "\n";
        });

    };

    template<typename T, size_t ...N>
    inline std::ostream& operator<<( std::ostream &stream, const parameter_decompo_t<T,N...> &val ) {
        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "DIAG(" << I << ")\n" << std::get<I>( val.diagonals ) << "\n";
        });

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            stream << "SUB(" << I << ")\n" << std::get<I>( val.subdiagonals ) << "\n";
        });

        return stream;
    }






    template<typename T, size_t ...N>
    inline typename decompos_diagentry<T, N...>::type compute_diagonals(
            const parameter_decompo_t<T,N...> &var ) {

        typename decompos_diagentry<T, N...>::type res;
        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            auto& L = std::get<I>( var.subdiagonals );
            auto& D = std::get<I>( var.diagonals );

            std::get<I>(res) = L * blaze::trans(L)
                                + D * blaze::trans(D);
        });

        return std::move(res);
    }

    template<typename T, size_t ...N>
    inline typename decompos_diagentry<T, N...>::type compute_residual(
            const typename decompos_diagentry<T, N...>::type &tparam) {

        typename decompos_diagentry<T, N...>::type res;

        std::for_range<0,sizeof... (N)-2>([&]<auto I>(){
            auto& param = std::get<I>( tparam );
            std::get<I>( res ) = param;
            blaze::diagonal( std::get<I>( res ) ) = 0.0;
        });

        std::get<sizeof... (N)-2>(res) = std::get<sizeof... (N)-2>(tparam)
                - blaze::IdentityMatrix<T>( at<sizeof... (N)-1,N...>() );

        return std::move( res );
    }

    template<typename T, size_t ...N>
    inline T compute_loss( const typename decompos_diagentry<T, N...>::type &tparam,
                   const typename decompos_diagentry<T, N...>::type &dual,
                   const parameter_decompo_t<T,N...> &val, const T lipschitz, const T gamma,
                   const typename network_topology<T,N...>::type &weights ) {

        T loss = 0;
        std::for_range<0,sizeof... (N)-2>([&]<auto I>(){
           typedef  blaze::StaticMatrix<T, at<I+1,N...>(),
                  at<I+1,N...>(), blaze::rowMajor> matrix_t;
           typedef  blaze::StaticMatrix<T, at<I+1,N...>(),
                  at<I,N...>(), blaze::rowMajor> matrixl_t;

            auto& param = std::get<I>( tparam );
            auto& wbar = std::get<I>( weights ).weight;
            auto& L = std::get<I>( val.subdiagonals );

            matrixl_t norm;

            if constexpr ( I == 0 )
                norm = param*wbar + 2*L*lipschitz;
            else norm = param*wbar + 2*L*blaze::trans( std::get<I-1>( val.diagonals ));

            matrix_t tmp3 = std::get<I>( dual ) % param;
            blaze::diagonal( tmp3 ) = 0.0;

            matrix_t tmp4 = param;
            blaze::diagonal( tmp4 ) = 0.0;

            loss += blaze::sum( blaze::pow( norm ,2) )
                     + blaze::sum( tmp3 )
                     + 0.5*gamma*blaze::sum( blaze::pow( tmp4 ,2));

        });

        auto tmp = std::get<sizeof... (N)-2>(tparam) - blaze::IdentityMatrix<T>( at<sizeof... (N)-1,N...>() );

        loss += blaze::sum( blaze::pow( std::get<sizeof... (N)-2>( val.subdiagonals ) * blaze::trans(  std::get<sizeof... (N)-3>( val.diagonals ) ) +
                        std::get<sizeof... (N)-2>( weights ).weight , 2 ))
                + blaze::sum(  std::get<sizeof... (N)-2>( dual ) % tmp )
                + 0.5*gamma*blaze::sum( blaze::pow( tmp , 2 ) );

        return loss;
    }




    template<size_t INDEX, typename T, size_t ...N>
    inline std::tuple<blaze::StaticMatrix<T,at<INDEX+1,N...>(),at<INDEX+1,N...>(), blaze::rowMajor>,
    blaze::StaticMatrix<T,at<INDEX+1,N...>(),at<INDEX,N...>(), blaze::rowMajor>>
            compute_gradient_wrt_index( const typename decompos_diagentry<T, N...>::type &tparam,
                                const typename decompos_diagentry<T, N...>::type &dual,
                                const parameter_decompo_t<T,N...> &val, const T lipschitz, const T gamma,
                                const typename network_topology<T,N...>::type &weights)  {

        static_assert ( INDEX < sizeof... (N)-1 , "index out of range" );

        typedef std::integral_constant<size_t, sizeof... (N)-2> L;
        typedef  blaze::StaticMatrix<T,at<INDEX+1,N...>(),
                at<INDEX+1,N...>(), blaze::rowMajor> matrix_t;

        typedef  blaze::StaticMatrix<T,at<INDEX+1,N...>(),
                at<INDEX,N...>(), blaze::rowMajor> matrixl_t;

        if constexpr( INDEX < L::value  ) {
            auto& param = std::get<INDEX>( tparam );
            auto& wbar = std::get<INDEX>( weights ).weight;
            auto& L = std::get<INDEX>( val.subdiagonals );

            auto& paramp = std::get<INDEX+1>( tparam );
            auto& wbarp = std::get<INDEX+1>( weights ).weight;
            auto& Lp = std::get<INDEX+1>( val.subdiagonals );

            auto& D = std::get<INDEX>( val.diagonals );

            matrixl_t norm;
            if constexpr ( INDEX == 0 )
                norm = 2*( param*wbar + 2*lipschitz*L );
            else norm = 2*( param*wbar + 2*std::get<INDEX-1>(
                                      val.diagonals )*L );

            matrix_t tmp1 = norm*blaze::trans(wbar) , tmp2;
            matrix_t tmp4 = 2*gamma*param;
            blaze::diagonal( tmp4 ) = 0.0;

            if constexpr ( INDEX == L::value-1 ) {
                tmp2 = 2*( blaze::trans(wbarp) + D*blaze::trans(Lp) )*Lp;
            } else {
                tmp2 = 4*( blaze::trans(wbarp)*paramp + 2*D*blaze::trans(Lp) )*Lp;
            }

            matrix_t tmp3 = std::get<INDEX>( dual );
            blaze::diagonal( tmp3 ) = 0.0;

            matrixl_t tmp5;
            if constexpr ( INDEX == 0 ) tmp5 = 2*norm*lipschitz;
            else tmp5 = 2*norm*std::get<INDEX-1>( val.diagonals );

            auto gradd = (tmp1 + blaze::trans(tmp1)
                     + tmp3 + blaze::trans(tmp3)
                     + tmp4 )*D + tmp2;

            auto gradl = (tmp1 + blaze::trans(tmp1)
                     + tmp3 + blaze::trans(tmp3)
                     + tmp4 )*L + tmp5;

            return std::make_tuple( std::move(gradd),
                                    std::move(gradl) );
        }

        /*if constexpr ( INDEX > 0 && INDEX < L::value ) {
            static_assert ( INDEX == 0 , "have to be implemented");
        }*/

        if constexpr ( INDEX == L::value ) {
            auto& wbar = std::get<INDEX>( weights ).weight;
            auto& param = std::get<INDEX>( tparam );
            auto& D = std::get<INDEX>( val.diagonals );
            auto& L = std::get<INDEX>( val.subdiagonals );

            auto& Dm = std::get<INDEX-1>( val.diagonals );

            auto tmp1 =  std::get<INDEX>( dual ) + blaze::trans(std::get<INDEX>( dual ))
                + 2*gamma*(param - blaze::IdentityMatrix<T>( at<L::value+1, N...>() ));

            auto gradd = tmp1*D;

            auto gradl = tmp1*L + 2*( wbar + L*blaze::trans(Dm) )*Dm;

            return std::make_tuple( std::move(gradd),
                                    std::move(gradl) );
        }

    }



    template<typename T, size_t ...N>
    inline void extract_weights( const typename decompos_diagentry<T, N...>::type &tparam,
                                 const parameter_decompo_t<T,N...> &val, const T lipschitz,
                                 typename network_topology<T,N...>::type &weights ) {

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            typedef  blaze::StaticMatrix<T, at<I+1,N...>(),
                      at<I,N...>(), blaze::rowMajor> matrixl_t;

            matrixl_t weighthat;
            if constexpr ( I == 0 )
                weighthat = -1.0*lipschitz * std::get<I>( val.subdiagonals );
            else weighthat = -1.0*std::get<I>( val.subdiagonals )
                * blaze::trans( std::get<I-1>( val.diagonals ) );

            if constexpr ( I >= sizeof... (N)-2 )
                 std::get<I>( weights ).weight =  blaze::solve(
                         blaze::decldiag( std::get<I>(tparam) ) , weighthat );
            else std::get<I>( weights ).weight =  blaze::solve(
                         blaze::decldiag( 0.5*std::get<I>(tparam) ) , weighthat );
        });


    }



    /*template<typename T, size_t ...N>
    inline auto compute_variable( const parameter_decompo_t<T,N...> &val ) {
        parameter_decompo_t<T,N...> result;

        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            auto &curr = std::get<I>( result.diagonals );
            auto &diag = std::get<I>(val.diagonals);

            if constexpr ( I == 0 )
                curr = blaze::decllow( diag * blaze::trans( diag ) );


            if constexpr ( I > 0) {
                 auto& sub =  std::get<I-1>(val.subdiagonals);

                 curr = blaze::decllow( diag * blaze::trans( diag )
                            + sub * blaze::trans( sub ) );
             }

        });

        std::for_range<0,sizeof... (N)-1>([&]<auto I>(){
            auto &curr = std::get<I>( result.subdiagonals );
            auto &diag = std::get<I>(val.diagonals);
            auto &sub = std::get<I>(val.subdiagonals);

                                              print_dims( "sub", sub);
                                              print_dims( "diag", diag);

            curr = sub * blaze::trans( diag );
        });

        return std::move(result);
    }




    template<typename T, size_t ...N>
    inline auto compute_residual( const T &lipschitz, const parameter_decompo_t<T,N...> &val ) {
        parameter_decompo_dual_t<T,N...> result;

        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            auto &curr = std::get<I>( result.diagonals );
            auto &diag = std::get<I>(val.diagonals);

            if constexpr ( I == 0 ) {
                curr = blaze::decllow( diag * blaze::trans( diag ) );
                blaze::diagonal( curr ) -= std::pow( lipschitz, 2);
            }


            if constexpr ( I > 0  && I+1 < sizeof... (N)) {
                 auto& sub =  std::get<I-1>(val.subdiagonals);

                 curr = blaze::decllow( diag * blaze::trans( diag )
                           + sub * blaze::trans( sub ) );

                 blaze::diagonal( curr ) = blaze::uniform( at<I,N...>() , 0.0 );
             }


             if constexpr ( I > 0  && I+1 < sizeof... (N)) {
                auto& sub =  std::get<I-1>(val.subdiagonals);
                curr = blaze::decllow( diag * blaze::trans( diag )
                           + sub * blaze::trans( sub ) );
                blaze::diagonal( curr ) -= 1.0;
             }


        });

        return std::move(result);
    }





    template<typename T, size_t ...N>
    inline T compute_trace( const parameter_decompo_dual_t<T,N...> &dual,
                               const parameter_decompo_t<T,N...> &val ) {

        T sum = 0.0;

        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            auto &d = std::get<I>( dual.diagonals );
            auto &v = std::get<I>( val.diagonals );

            sum += blaze::sum( d % v );
        });

        return std::move(sum);

    }


    template<typename T, size_t ...N>
    inline T compute_norm( const parameter_decompo_dual_t<T,N...> &dual ) {

        T sum = 0.0;

        std::for_range<0,sizeof... (N)>([&]<auto I>(){
            auto &d = std::get<I>( dual.diagonals );

            sum += blaze::sum( blaze::pow( d, 2) );
        });

        return std::move(sum);

    }*/








    template<typename T, size_t ...N>
    struct prod_t<T, parameter_decompo_t<T,N...>, parameter_decompo_t<T,N...> > {
        static inline T inner( const parameter_decompo_t<T,N...> &m1,
                               const parameter_decompo_t<T,N...> &m2) {
            typedef  decltype ( parameter_decompo_t<T,N...>::diagonals ) A1;
            typedef  decltype ( parameter_decompo_t<T,N...>::subdiagonals ) A2;

            return prod_t<T,A1,A1>::inner( m1.diagonals, m1.diagonals )
                    + prod_t<T,A2,A2>::inner( m1.subdiagonals, m1.subdiagonals ) ;
        }
    };




}

#endif // __LIPNET_DECOMPOSITION_LIP_HPP__
