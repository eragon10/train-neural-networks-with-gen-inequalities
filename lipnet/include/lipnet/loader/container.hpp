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

#ifndef __LIPNET_DATA_CONTAINER_HPP__
#define __LIPNET_DATA_CONTAINER_HPP__

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


#include <cereal/cereal.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>




namespace lipnet {

    /**
     * @brief trining data holder; data_container_t
     * @tparam T numerical value type
     */

    template<typename T>
    struct data_container_t {

        using matrix_t = blaze::DynamicMatrix<T, blaze::rowMajor>;


        template<bool saveing> struct view_t {
            using refer_t = decltype( blaze::row( std::declval<typename std::conditional<saveing,
                        const matrix_t, matrix_t>::type>(), std::declval<int>() ) ); refer_t value;
            using item_t = typename std::conditional<saveing, const T, T>::type;

            template<class Archive> void serialize(Archive &ar)
                { cereal::size_type size = value.size();
                  ar(  cereal::make_size_tag( size ) );
                  for( size_t i = 0UL; i < size; i++ ) ar( value[i] ); }
        };

        template<bool saveing = true> struct tuple_t { view_t<saveing> x, y;
            template<class Archive> void serialize(Archive &ar)
                { ar(  cereal::make_nvp("x", x ),
                       cereal::make_nvp("y", y ) ); }};

        template<bool saveing = true> struct data_t {
            using value_t = typename std::conditional<saveing,
                            const matrix_t, matrix_t>::type;
            value_t &x, &y;
            template<class Archive> void serialize(Archive &ar)
                { cereal::size_type number = x.rows();
                  ar( cereal::make_size_tag(number) );
                  for( size_t i = 0; i < number; i++ ) {
                    ar( tuple_t<saveing> { view_t<saveing>{blaze::row(x,i)},
                                           view_t<saveing>{blaze::row(y,i)} } );
                  } }};


        matrix_t x, y;
    };


    /**
     * @brief serialize data_container_t;
     * @cite cereallib
     */
    template<class Archive, typename T>
    void save(Archive & archive, const data_container_t<T> &m)
    {
        typedef blaze::StaticVector<int,3,blaze::columnVector> vec_t;
        typedef typename data_container_t<T>::template
                data_t<true> serializable_t;

        assert( m.x.columns() == m.y.columns() );

        vec_t dims = {  m.x.rows(),  m.x.columns(), m.y.columns()  };
        archive( cereal::make_nvp("size", dims ) );

        archive( cereal::make_nvp("data",
                    serializable_t{ m.x, m.y } ) );
    }

    /**
     * @brief seserialize data_container_t
     * @cite cereallib
     */

    template<class Archive, typename T>
    void load(Archive & archive, data_container_t<T> &m)
    {
        typedef blaze::StaticVector<int,3,blaze::columnVector> vec_t;
        typedef typename data_container_t<T>::template
                data_t<false> serializable_t;

        vec_t dims;
        archive( cereal::make_nvp( "size", dims ) );

        blaze::resize( m.x, dims[0], dims[1] );
        blaze::resize( m.y, dims[0], dims[2] );


        archive( cereal::make_nvp( "data",
                    serializable_t{ m.x, m.y } ) );
    }


}

#endif // __LIPNET_DATA_CONTAINER_HPP__
