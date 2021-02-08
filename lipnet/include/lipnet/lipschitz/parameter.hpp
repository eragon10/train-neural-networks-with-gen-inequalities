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

#ifndef __LIPNET_LIPSCHITZ_PARAMETER_HPP__
#define __LIPNET_LIPSCHITZ_PARAMETER_HPP__

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

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>

#include <lipnet/lipschitz/structure.hpp>





namespace lipnet {

    template<typename T, size_t ...N>
    struct lipcalc_parameter_t {
        T rho;  blaze::StaticVector<T, sum<sizeof... (N)-1,N...>() - at<0,N...>() > tmat;
    };



    template<typename T, size_t ...N>
    lipcalc_parameter_t<T,N...> operator+( const lipcalc_parameter_t<T,N...> &self,
                                           const lipcalc_parameter_t<T,N...> &other ) {
        return lipcalc_parameter_t<T,N...>{ self.rho + other.rho, self.tmat + other.tmat };
    }

    template<typename T, size_t ...N>
    lipcalc_parameter_t<T,N...> operator-( const lipcalc_parameter_t<T,N...> &self,
                                           const lipcalc_parameter_t<T,N...> &other ) {
        return lipcalc_parameter_t<T,N...>{ self.rho - other.rho, self.tmat - other.tmat };
    }




    template<typename T, size_t ...N>
    lipcalc_parameter_t<T,N...>& operator+=( lipcalc_parameter_t<T,N...> &self,
                                             const lipcalc_parameter_t<T,N...> &other ) {
        self.rho += other.rho;
        self.tmat += other.tmat;
        return self;
    }


    template<typename T, size_t ...N>
    lipcalc_parameter_t<T,N...>& operator-=( lipcalc_parameter_t<T,N...> &self, const
                                             lipcalc_parameter_t<T,N...> &other ) {
        self.rho -= other.rho;
        self.tmat -= other.tmat;
        return self;
    }




    template<typename T, size_t ...N>
    lipcalc_parameter_t<T,N...> operator*( const T &var, const lipcalc_parameter_t<T,N...> &other ) {
        return lipcalc_parameter_t<T,N...>{ var * other.rho, var * other.tmat } ;
    }











    template<typename T, size_t ...N>
    struct norm_t<T, lipcalc_parameter_t<T,N...>> {
        static inline T norm( const lipcalc_parameter_t<T,N...> &m ) {
            return std::abs( m.rho ) + blaze::norm( m.tmat ) ;
        }
    };


    template<typename T, size_t ...N>
    struct prod_t<T, lipcalc_parameter_t<T,N...>, lipcalc_parameter_t<T,N...> > {
        static inline T inner( const lipcalc_parameter_t<T,N...> &m1,
                               const lipcalc_parameter_t<T,N...> &m2) {
            return m1.rho*m2.rho + blaze::inner( m1.tmat, m2.tmat ) ;
        }
    };



}

#endif // __LIPNET_LIPSCHITZ_PARAMETER_HPP__
