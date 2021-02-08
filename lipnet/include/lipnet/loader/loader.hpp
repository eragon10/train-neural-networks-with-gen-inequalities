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

#ifndef __LIPNET_LOADER_HPP__
#define __LIPNET_LOADER_HPP__

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

#include <csv2/csv2.hpp>

#include <cereal/cereal.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>




namespace lipnet {

    /**
     * @brief create one hot vectors from lables; labels must
     *          be natural numbers starting from zero
     *
     * @tparam T numerical value type
     */

    template<typename T>
    blaze::DynamicMatrix<T, blaze::rowMajor> make_one_hot( const blaze::DynamicVector<T,
                                                           blaze::columnVector>&tens , const size_t CL) {

        blaze::DynamicMatrix<T, blaze::rowMajor> res( CL,  tens.size(), 0 );

        for( int i = 0; i < tens.size() ; i++ ) {
            T value = tens.at(i);
            res( (int) value, i ) = 1;
        }

        return std::move(res);
    }







    /**
     * @brief  struct for loading matrix from csv file;
     * @tparam T numerical value type
     * @cite csv2lib
     */

    template<typename T>
    struct loader_t
    {

        typedef blaze::DynamicMatrix<T, blaze::rowMajor> dmatrix_t;

        /**
         * @brief load matrix from csv file;
         * @param path path to file on filesystem
         * @return matrix
         */

        static std::optional<dmatrix_t> load( const std::string &path ) {
            csv2::Reader<csv2::delimiter<','>,
                         csv2::quote_character<'"'>,
                         csv2::first_row_is_header<false>,
                         csv2::trim_policy::trim_whitespace> csv;

            if( !csv.mmap( path ) ) return std::nullopt;


            const auto header = csv.header();

            dmatrix_t data( csv.cols(), csv.rows() );

            size_t row_counter = 0;
            for (const auto row: csv) {
                size_t col_counter = 0;
                for(const auto cell: row) {
                    std::string value;
                    std::string::size_type sz;

                    cell.read_value( value  );
                    data(col_counter, row_counter)
                            = std::stod( value, &sz);
                     col_counter++;
                }

                row_counter++;
            }


            return std::move(data);
        }

    };

}

#endif // __LIPNET_LOADER_HPP__
