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

#ifndef __LIPNET_NETWORK_HPP__
#define __LIPNET_NETWORK_HPP__

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
#include <lipnet/problem.hpp>

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/activation.hpp>
#include <lipnet/network/topology.hpp>





namespace lipnet {

    template<size_t I, size_t... Ints>
    constexpr size_t at() {
        constexpr size_t arr[] = {Ints...};
        return arr[I];
    }


    template<size_t... Ints, size_t... Seq>
    constexpr size_t sum_(std::integer_sequence<size_t, Ints...>,
                          std::integer_sequence<size_t, Seq...> ) {
        if constexpr( sizeof... (Seq) == 0) {
                return 0;
        } else {
            constexpr size_t arr[] = {Ints...};
            return (arr[Seq] + ... );
        }
    }

    template<size_t I, size_t... Ints>
    constexpr size_t sum() {
        return sum_(  std::integer_sequence<size_t, Ints...>{},
                      std::make_integer_sequence<size_t,I>{});
    }


    /**
     * @brief The network_t struct; neural network implementation
     * @tparam T numerical value type
     * @tparam ATYPE activation function type
     * @tparam N network topology
     */
    template<typename T, template<typename> typename ATYPE, size_t ...N>
    struct network_t {

        typedef typename network_topology<T, N...>::type layer_t;

        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;

        typedef blaze::StaticVector<T, at<L::value, N...>(),
                        blaze::columnVector> outvec_t;
        typedef blaze::StaticVector<T, at<0, N...>(),
                        blaze::columnVector> invec_t;


        /// serialization helper struct
        struct topology_serialization_t {
            template<class Archive> void serialize(Archive &ar)
                { cereal::size_type number = NL::value;
                  ar( cereal::make_size_tag(number), N... ); }};

        /// serialization helper struct
        template<bool saveing = true> struct data_serialization_t {
            using value_t = typename std::conditional<saveing,
                            const layer_t, layer_t>::type; value_t &layersdata;
            using seq_t = std::make_integer_sequence<size_t,L::value>;

            template<class Archive, size_t ...INTS> void serialize_impl( Archive &ar,
                     const std::integer_sequence<size_t, INTS...>& ) {
                 ar( std::get<INTS>( layersdata ) ... ); }

            template<class Archive> void serialize(Archive &ar)
                { cereal::size_type number = L::value;
                  ar( cereal::make_size_tag(number) );
                  if constexpr ( !saveing ) assert( number == L::value );
                  serialize_impl( ar, seq_t{}); }};


        /// weights and biases
        layer_t layers;

        /**
         * @brief query the neural network
         *                  @f[  z_l = W_l x_l \quad x_{l+1} = \sigma(z_l) \quad \cdots  @f]
         * @param input vector
         * @return output vector
         */

        outvec_t query( const invec_t& input ) const {
            typedef typename generate_data<T, N...>::type xdata_t; xdata_t x;

            std::get<0>( x ) = input;

            std::for_range<0,L::value-1>([&]<auto I>(){
                auto& layer = std::get<I>(layers);
                std::get<I+1>(x) = ATYPE<T>::template forward<at<I+1,N...>(),1>(
                                        layer.weight * std::get<I>(x) + layer.bias );
            });

            auto& layer = std::get<L::value-1>(layers);
            return layer.weight * std::get<L::value-1>(x) + layer.bias;

        }



        /// serialize network
        template <class Archive>
        void save( Archive & ar) const
        {
            typedef data_serialization_t<true> serilizable_t;
            typedef topology_serialization_t meta_serilizable_t;
            ar( cereal::make_nvp("topology", meta_serilizable_t{} ) );
            ar( cereal::make_nvp("data", serilizable_t{layers} ) );
        }

        /// deserialize network
        template <class Archive>
        void load( Archive & ar )
        {
            typedef data_serialization_t<false> serilizable_t;
            constexpr std::array<size_t, sizeof... (N)>
                    realtopt = { N... };

            std::array<size_t, sizeof... (N)> topo;
            ar( cereal::make_nvp("topology", topo) );

            for(int i = 0; i < topo.size(); i++)
              if( topo[i] != realtopt[i] )
                  throw std::string{"wrong topology"};

            ar( cereal::make_nvp("data", serilizable_t{layers} ) );
        }

        
    };

}

#endif // __LIPNET_NETWORK_HPP__
