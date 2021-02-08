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

#ifndef __LIPBET_LAYER_HPP__
#define __LIPBET_LAYER_HPP__

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


#include <cereal/cereal.hpp>

#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>


#include <lipnet/traits.hpp>
#include <lipnet/tensor.hpp>
#include <lipnet/variable.hpp>



namespace lipnet {

    /**
     * @brief The make_random function; initilize vector with random values; uniform distribution
     * @tparam T numerical value type
     * @tparam Iter container iteration type
     * @param start begin iterator
     * @param end end iterator
     * @param min uniform distribution parameter
     * @param max uniform distribution parameter
     * @param n uniform distribution parameter
     */

    template< typename T, class Iter >
    void make_random( Iter start, Iter end, const T &min, const T &max, size_t n = 5000)
    {
        static std::random_device rd;
        static std::mt19937 mte(rd());

        std::uniform_int_distribution<int> dist(1, n);

        std::generate(start, end, [&] () {
            return min + (max-min)/n*dist(mte); });
    }




    /**
     * @brief The layer_t struct; the layer implementation of each layer;
     *        contains the weight and the biases
     * @tparam T numerical value type
     * @tparam I input dimension
     * @tparam O output dimension
     */

    template<typename T, size_t I, size_t O>
    struct layer_t {

        typedef std::array<T, I*O> weight_array_t;
        typedef std::array<T, O> bias_array_t;

        typedef blaze::StaticMatrix<T,O,I, blaze::columnMajor> MT;
        typedef blaze::StaticVector<T,O,blaze::columnVector> VT;

        MT weight; VT bias;

        inline layer_t( MT&& w, VT&& b)
            : weight{ std::move(w) },  bias{ std::move(b) } {}

        explicit layer_t() : weight{0}, bias{0} {}

        /**
         * @brief The layer_t constructor; initilize weight and bias with random values
         * @param var some kind of variance
         */
        explicit layer_t(const T &var) {
            std::array<T,O> binit; std::array<std::array<T,I>,O> winit;
            make_random( std::begin(binit), std::end(binit), -var, var);
            std::for_each( std::begin(winit), std::end(winit), [&var](auto &&val){
                 make_random( std::begin(val), std::end(val), -var, var);
            });

            bias =  VT(binit);   weight = MT(winit);
        }


        /// @brief serialize layer_t
        template <class Archive>
        void serialize( Archive & ar)
        {
            ar( cereal::make_nvp("weight", weight ) );
            ar( cereal::make_nvp("bias", bias ) );
        }



    };


    
    
    
    




    template<typename T, size_t I, size_t O>
    inline layer_t<T,I,O>& operator-=(layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        a.weight -= b.weight; a.bias -= b.bias;
        return a;
    }

    template<typename T, size_t I, size_t O>
    inline layer_t<T,I,O>& operator+=(layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        a.weight += b.weight; a.bias += b.bias;
        return a;
    }



    template<typename T, size_t I, size_t O>
    inline auto operator*(const T&a, const layer_t<T,I,O> &b) {
        return layer_t<T,I,O>( a*b.weight, a*b.bias );
    }

    template<typename T, size_t I, size_t O>
    inline auto operator+(const T&a, const layer_t<T,I,O> &b) {
        return layer_t<T,I,O>( a+b.weight, a+b.bias );
    }



    template<typename T, size_t I, size_t O>
    inline auto operator*(const layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        return layer_t<T,I,O>( a.weight % b.weight, a.bias % b.bias );
    }

    template<typename T, size_t I, size_t O>
    inline auto operator/(const layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        typename  layer_t<T,I,O>::MT tmp = 1 / b.weight;
        return layer_t<T,I,O>( a.weight % tmp , a.bias / b.bias );
    }

    template<typename T, size_t I, size_t O>
    inline auto operator-(const layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        return layer_t<T,I,O>( a.weight-b.weight, a.bias-b.bias );
    }

    template<typename T, size_t I, size_t O>
    inline auto operator+(const layer_t<T,I,O> &a, const layer_t<T,I,O> &b) {
        return layer_t<T,I,O>( a.weight+b.weight, a.bias+b.bias );
    }







    template<typename T, size_t I, size_t O>
    struct generator_t<layer_t<T,I,O>> {
        static inline layer_t<T,I,O> make(T val) {
            return layer_t<T,I,O>( val );
        }
    };


    template<typename T, size_t I, size_t O>
    struct norm_t<T, layer_t<T,I,O>> {
        static inline T norm( const layer_t<T,I,O> &m ) {
            return blaze::norm( m.weight ) + blaze::norm( m.bias );
        }
    };


    template<typename T, size_t I, size_t O>
    struct function_t<layer_t<T,I,O>> {
        static inline auto square( const layer_t<T,I,O> &m ) {
            return layer_t<T,I,O>( blaze::pow(m.weight,2),  blaze::pow(m.bias,2) );
        }

        static inline auto sqrt( const layer_t<T,I,O> &m ) {
            return layer_t<T,I,O>( blaze::sqrt(m.weight) , blaze::sqrt(m.bias) );
        }
    };


    template<typename T, size_t I1, size_t O1, size_t I2, size_t O2>
    struct prod_t<T, layer_t<T,I1,O1>, layer_t<T,I2,O2>> {
        static inline T inner( const layer_t<T,I1,O1> &m1, const layer_t<T,I2,O2> &m2 ) {
            return blaze::inner( m1.weight , m2.weight ) + blaze::inner( m1.bias, m2.bias);
        }
    };


}

#endif // __NN_LAYER_HPP__
