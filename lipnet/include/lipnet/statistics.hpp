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

#ifndef __LIPNET_STATISTICS_HPP__
#define __LIPNET_STATISTICS_HPP__

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <list>
#include <tuple>
#include <functional>
#include <map>
#include <algorithm>
#include <chrono>
#include <utility>
#include <initializer_list>
#include <deque>

#include <lipnet/traits.hpp>

#include <cereal/cereal.hpp>

namespace lipnet {


    /**
     * @brief The series_t struct. Base struct for logging.
     * @tparam T numerical type
     */

    template<typename T>
    struct series_t {
        std::vector<T> data;

        explicit series_t( const size_t size = 0 ) {
            data.reserve( size );
        }


        T& operator()( const size_t index ) {
            return data.at(index);
        }

        series_t<T>& operator<<( const T point ) {
            data.push_back( point );
            return *this;
        }
    };



    /// serialize series_t type
    template<class Archive, typename T>
    void save(Archive & archive, const series_t<T> & s )
    {
        archive( cereal::make_size_tag( static_cast<cereal::size_type>(s.data.size())) );
        std::for_each( std::begin(s.data), std::end(s.data), [&](auto &val){
            archive( val );
        });
    }

    /// deserialize series_t type
    template<class Archive, typename T>
    void load(Archive & archive, series_t<T> & s )
    {
        cereal::size_type size;
        archive( cereal::make_size_tag(size) );
        s.data.resize( size );

        for( int i=0; i < size; i++)
            archive(s.data[i]);
    }




    /**
     * @brief The statistics_helper struct. Helper function to disable
     *        logging for performence reasons if it is desired.
     */
    struct statistics_helper {
        template<class TT> struct void_t { typedef void type; };

        template<class TT, class U = void>
        struct stats_type_exists { enum { value = 0 }; typedef std::void_type type; };

        template<class TT>
        struct stats_type_exists<TT, typename
                void_t<typename TT::statistics_t>::type> { enum { value = 1 };
                   typedef typename TT::statistics_t type; };
    };


}

#endif // __LIPNET_STATISTICS_HPP__
