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

#ifndef __LIPNET_TENSOR_HPP__
#define __LIPNET_TENSOR_HPP__

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <initializer_list>
#include <deque>


#include <blaze/Blaze.h>

#include <lipnet/traits.hpp>
#include <lipnet/variable.hpp>


namespace lipnet {

    

    /**
     * @brief The prod_t struct for blaze::StaticVector.
     * @tparam T numerical value type
     * @tparam N1 dimension of first argument
     * @tparam N2 dimension of second argument
     * @see lipnet::prod_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2>
    struct prod_t<T, blaze::StaticVector<T,N1,blaze::columnVector>,
                     blaze::StaticVector<T,N2,blaze::columnVector> > {

        /**
         * @brief The inner method. Implemention of the inner
         *        product of blaze::StaticVector type. \f$ m_1^\top m_2 \f$
         * @param m1 first argument (blaze::StaticVector<T,N1,blaze::columnVector>)
         * @param m2 second argument (blaze::StaticVector<T,N2,blaze::columnVector>)
         * @return inner product of m1 and m2
         */
        static inline T inner( const blaze::StaticVector<T,N1,blaze::columnVector> &m1,
                               const blaze::StaticVector<T,N2,blaze::columnVector> &m2 ) {
            return blaze::inner( m1, m2 );
        }

        /**
         * @brief The outer method. Implemention of the outer
         *        product of blaze::StaticVector type. \f$ m_1 m_2^\top \f$
         * @param m1 first argument (blaze::StaticVector<T,N1,blaze::columnVector>)
         * @param m2 second argument (blaze::StaticVector<T,N2,blaze::columnVector>)
         * @return outer product of m1 and m2
         */
        static inline auto outer( const blaze::StaticVector<T,N1,blaze::columnVector> &m1,
                                  const blaze::StaticVector<T,N2,blaze::columnVector> &m2 ) {
            return blaze::outer( m1, m2 );
        }

    };




    /**
     * @brief The prod_t struct for blaze::StaticMatrix.
     * @tparam T numerical value type
     * @tparam N1 row dimension of first argument
     * @tparam N2 column dimension of first argument
     * @tparam N3 row dimension of second argument
     * @tparam N4 column dimension of second argument
     * @see lipnet::prod_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2, size_t N3, size_t N4>
    struct prod_t<T, blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>,
                     blaze::StaticMatrix<T,N3,N4,blaze::rowMajor> > {

        /**
         * @brief The inner method. Implemention of the inner
         *        product of blaze::StaticVector type. \f$ m_1^\top m_2 \f$
         * @param m1 first argument (blaze::StaticVector<T,N1,blaze::columnVector>)
         * @param m2 second argument (blaze::StaticVector<T,N2,blaze::columnVector>)
         * @return inner product of m1 and m2
         */
        static inline T inner( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &m1,
                               const blaze::StaticMatrix<T,N3,N4,blaze::rowMajor> &m2 ) {
            return blaze::inner( m1, m2 );
        }

        /**
         * @brief The outer method. Implemention of the outer
         *        product of blaze::StaticMatrix type.
         * @param m1 first argument (blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>)
         * @param m2 second argument (blaze::StaticMatrix<T,N3,N4,blaze::rowMajor>)
         * @return kronecker product of m1 and m2
         */
        static inline auto outer( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &m1,
                                  const blaze::StaticMatrix<T,N3,N4,blaze::rowMajor> &m2 ) {
            return blaze::kron( m1, m2 );
        }

    };


    /**
     * @brief The equation_system_t struct for blaze::StaticMatrix.
     * @tparam T numerical value type
     * @tparam N1 row dimension of first argument
     * @tparam N2 column dimension of first argument
     * @tparam N3 row dimension of second argument
     * @tparam N4 column dimension of second argument
     * @see lipnet::equation_system_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2, size_t N3, size_t N4>
    struct equation_system_t< blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>,
                              blaze::StaticMatrix<T,N3,N4,blaze::rowMajor> > {

        /**
         * @brief The solve method. Solve system of equations. \f$ A X = A \f$
         * @param A matrix \f$ A \f$ (first argument)
         * @param B matrix \f$ A \f$ (second argument)
         * @return matrix \f$ X \f$
         */
        static inline auto solve( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &A,
                                  const blaze::StaticMatrix<T,N3,N4,blaze::rowMajor> &B ) {
            return blaze::solve( A, B );
        }
    };



    /**
     * @brief The equation_system_t struct for blaze::StaticMatrix and blaze::StaticVector.
     * @tparam T numerical value type
     * @tparam N1 row dimension of first argument
     * @tparam N2 column dimension of first argument
     * @tparam N3 dimension of second argument
     * @see lipnet::equation_system_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2, size_t N3>
    struct equation_system_t< blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>,
                              blaze::StaticVector<T,N3,blaze::columnVector> > {

        /**
         * @brief The solve method. Solve system of equations. \f$ A x = b \f$
         * @param A matrix \f$ A \f$ (first argument)
         * @param B vector \f$ b \f$ (second argument)
         * @return vector \f$ x \f$
         */
        static inline auto solve( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &A,
                                  const blaze::StaticVector<T,N3,blaze::columnVector> &B ) {
            return blaze::solve( A, B );
        }
    };



    /**
     * @brief The generator_t struct for blaze::StaticMatrix.
     * @tparam T numerical value type
     * @tparam N1 row dimension of return matrix
     * @tparam N2 column dimension of return matrix
     * @see lipnet::generator_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2>
    struct generator_t< blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> > {

        /// uniform distribution constructor \f$ \sim \mathcal{U}(-val,val) \f$
        static inline auto make( const T &val ) {
            return blaze::uniform(N1, N2, val);
        }

        /// uniform distribution constructor \f$  \sim \mathcal{U}(-val,val) \f$
        static inline auto unifrom( const T &val ) {
            return blaze::uniform(N1, N2, val);
        }

        /// identity constructor \f$ I \f$
        static inline auto identity() {
            static_assert ( N1 == N2, "have to provide square matrix");
            return blaze::IdentityMatrix<T>(N1);
        }
    };


    /**
     * @brief The generator_t struct for blaze::StaticVector.
     * @tparam T numerical value type
     * @tparam N dimension of return vector
     * @see lipnet::generator_t
     * @cite balzelib
     */

    template<typename T, size_t N>
    struct generator_t< blaze::StaticVector<T,N,blaze::columnVector> > {

         /// uniform distribution constructor \f$ \sim \mathcal{U}(-val,val) \f$
        static inline auto make( const T &val ) {
            return blaze::uniform(N, val);
        }

         /// uniform distribution constructor \f$ \sim \mathcal{U}(-val,val) \f$
        static inline auto unifrom( const T &val ) {
            return blaze::uniform(N, val);
        }

    };




    /**
     * @brief The norm_t struct for blaze::StaticVector.
     * @tparam T numerical value type
     * @tparam N dimension of argument
     * @see lipnet::norm_t
     * @cite balzelib
     */

    template<typename T, size_t N >
    struct norm_t<T, blaze::StaticVector<T,N,blaze::columnVector> > {

        /**
         * @brief The norm method. Compute norm of vector m. \f$ ||m||_2 \f$
         * @param m input vector
         * @return norm of vector m
         */
        static inline T norm( const blaze::StaticVector<T,N,blaze::columnVector>  &m ) {
            return blaze::norm( m );
        }
    };



    /**
     * @brief The norm_t struct for blaze::StaticMatrix.
     * @tparam T numerical value type
     * @tparam N1 row dimension of argument
     * @tparam N2 column dimension of argument
     * @see lipnet::norm_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2 >
    struct norm_t<T, blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> > {

        /**
         * @brief The norm method. Compute norm of vector m. \f$ ||m||_{2\mathrm{-ind.}} \f$
         * @param m input matrix
         * @return norm of matrix m
         */
        static inline T norm( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>  &m ) {
            return blaze::norm( m );
        }
    };




    /**
     * @brief The function_t struct for blaze::StaticVector.
     * @tparam T numerical value type
     * @tparam N dimension of argument
     * @see lipnet::function_t
     * @cite balzelib
     */

    template<typename T, size_t N>
    struct function_t<blaze::StaticVector<T,N,blaze::columnVector>> {

        /// transpose vector \f$ v^\top \f$
        static inline auto trans( const blaze::StaticVector<T,N,blaze::columnVector> &vec ) {
            return blaze::trans(vec);
        }

        /// square vector elementwise
        static inline auto square( const blaze::StaticVector<T,N,blaze::columnVector> &vec ) {
            return blaze::pow( vec, 2 );
        }

        /// take square root of vector elementwise
        static inline auto sqrt( const blaze::StaticVector<T,N,blaze::columnVector> &vec ) {
            return blaze::sqrt( vec );
        }
    };




    /**
     * @brief The function_t struct for blaze::StaticMatrix.
     * @tparam T numerical value type
     * @tparam N1 row dimension of argument
     * @tparam N2 column dimension of argument
     * @see lipnet::function_t
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2>
    struct function_t<blaze::StaticMatrix<T,N1,N2,blaze::rowMajor>> {

        /// transpose matrix \f$ M^\top \f$
        static inline auto trans( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &m ) {
            return blaze::trans(m);
        }
    };


}













namespace blaze {


    /**
     * @brief The flatten function. Flatten matrix to vector.
     * @tparam T numerical value type
     * @tparam N1 row dimension of argument
     * @tparam N2 column dimension of argument
     * @return vector of dimension \f$ N_1 N_2 \f$
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2>
    inline auto flatten( const blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> &mat ) {
        typedef blaze::StaticVector<T,N1*N2,blaze::columnVector> vecor_t;

        vecor_t result;
        std::for_range<0,N1>([&]<auto I>(){
            blaze::subvector<I*N2,N2>( result )
                      = blaze::trans( blaze::row<I>( mat ) );
        });

        return std::move(result);
    }


    /**
     * @brief The tomatrix function. Reshape vector to matrix.
     *        Input vector is of dimension \f$ N_1 N_2 \f$.
     * @tparam T numerical value type
     * @tparam N1 row dimension of return matrix
     * @tparam N2 column dimension of return matrix
     * @return matrix
     * @cite balzelib
     */

    template<typename T, size_t N1, size_t N2>
    inline auto tomatrix( const blaze::StaticVector<T,N1*N2,blaze::columnVector> &vec ) {
        typedef blaze::StaticMatrix<T,N1,N2,blaze::rowMajor> matrix_t;

        matrix_t result;
        std::for_range<0,N1>([&]<auto I>(){
            blaze::row<I>( result ) = blaze::trans(
                     blaze::subvector<I*N2,N2>( vec ) );
        });

        return std::move(result);
    }

}



template<class Archive, typename T, size_t R, size_t C, bool SO>
void serialize(Archive & archive, blaze::StaticMatrix<T, R, C, SO> & m );

template<class Archive, typename T, size_t N>
void serialize(Archive & archive, blaze::StaticVector<T, N, blaze::columnVector>  & m );

template<class Archive, typename T, bool SO>
void serialize(Archive & archive, blaze::DynamicMatrix<T, SO> & m );

template<class Archive, typename T>
void serialize(Archive & archive, blaze::DynamicVector<T, blaze::columnVector>  & m );



#include <cereal/cereal.hpp>


/// @brief serialize blaze::StaticMatrix
/// @cite cereallib
/// @cite balzelib
template<class Archive, typename T, size_t R, size_t C, bool SO>
void serialize(Archive & archive, blaze::StaticMatrix<T, R, C, SO> & m )
{
    archive( cereal::make_size_tag(static_cast<cereal::size_type>(R*C)) );
    for( size_t i=0UL; i<m.rows(); ++i )
        for( size_t j=0UL; j<m.columns(); ++j ) {
            archive( m(i,j) );
        }
}

/// @brief serialize blaze::StaticVector
/// @cite cereallib
/// @cite balzelib
template<class Archive, typename T, size_t N>
void serialize(Archive & archive, blaze::StaticVector<T, N, blaze::columnVector>  & m )
{
    archive( cereal::make_size_tag(static_cast<cereal::size_type>(N)) );
    for( size_t i=0UL; i<m.size(); ++i ) {
        archive( m.at(i) );
    }
}






/// @brief serialize blaze::DynamicMatrix
/// @cite cereallib
/// @cite balzelib
template<class Archive, typename T, bool SO>
void serialize(Archive & archive, blaze::DynamicMatrix<T, SO> & m )
{
    archive( cereal::make_size_tag(static_cast<cereal::size_type>(
                                       m.rows() * m.columns() )) );
    for( size_t i=0UL; i<m.rows(); ++i )
        for( size_t j=0UL; j<m.columns(); ++j ) {
            archive( m(i,j) );
        }
}


/// @brief serialize blaze::DynamicVector
/// @cite cereallib
/// @cite balzelib
template<class Archive, typename T>
void serialize(Archive & archive, blaze::DynamicVector<T, blaze::columnVector>  & m )
{
    archive( cereal::make_size_tag(static_cast<cereal::size_type>( m.size() )) );
    for( size_t i=0UL; i<m.size(); ++i ) {
        archive( m.at(i) );
    }
}



#endif // __LIPNET_TENSOR_HPP__
