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

#ifndef __LIPNET_BACKPROPAGATION_HPP__
#define __LIPNET_BACKPROPAGATION_HPP__

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

#include <lipnet/network/layer.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/network.hpp>
#include <lipnet/network/activation.hpp>




namespace lipnet {

    /**
     * @brief The network_data_t struct; training dataset
     * @tparam T numerical value type
     * @tparam IN input dimension
     * @tparam OUT output dimension
     */

    template<typename T, size_t IN, size_t OUT>
    struct network_data_t {
        blaze::DynamicMatrix<T, blaze::rowMajor> idata, tdata;
    };


    /**
     * @brief The backpropagation_batch_t struct; implmentation of backtracking with batches
     * @tparam T numerical type
     * @tparam ATYPE activation function type
     * @tparam LOSS loss function type
     * @tparam BATCH batch size
     * @tparam N network topology
     */

    template<typename T, template<typename> typename ATYPE,
             template<typename> typename LOSS, size_t BATCH, size_t ...N>
    struct backpropagation_batch_t {


        template<size_t NN>
        using vector_t = blaze::StaticVector<T, NN, blaze::columnVector>;

        template<size_t NN1, size_t NN2>
        using matrix_t = blaze::StaticMatrix<T, NN1, NN2, blaze::rowMajor>;

        typedef std::integral_constant<size_t, sizeof... (N)-1> L;
        typedef std::integral_constant<size_t, (N + ... )> NL;
        typedef std::integer_sequence<size_t, N...> DIMS;


        typedef typename network_t<T,ATYPE, N...>::layer_t variable_t;

        typedef typename generate_batch_data_remove_first<T, BATCH, N...>::type zdata_t;
        typedef typename generate_batch_data<T, BATCH, N...>::type xdata_t;

        struct metainfo_t {
             size_t iter; metainfo_t() : iter{0} {}
        };

        network_data_t<T, at<0,N...>(), at<L::value,N...>() > training_data;
        LOSS<T> loss;

        explicit backpropagation_batch_t( LOSS<T>&& l, network_data_t<T, at<0,N...>(), at<L::value,N...>() > &&data )
            : training_data{ std::move(data) }, loss{ std::move(l) }  {

            if(  training_data.idata.rows() % BATCH != 0)
                throw std::string{"size not matching"};

        }

        /**
         * @brief run function; compute backpropagation
         * @param var current position
         * @param info optimisation metainfo which are needed during the iterations
         * @param gradient the computed gradients; the return value
         * @param objective the loss at the current position
         * @see compute( const variable_t& var, variable_t& gradient, T& objective ) const
         */

        void run( const variable_t& var, metainfo_t &info, variable_t& gradient, T& objective ) const {
            xdata_t xvec; zdata_t zvec; zdata_t deltavec;

            size_t i = (info.iter++) % (training_data.idata.rows() / BATCH);

            std::get<0>( xvec ) = blaze::trans( blaze::rows( training_data.idata,
                            [&i](size_t j){ return i*BATCH+j; }, BATCH ) );

            matrix_t<at<L::value, N...>(), BATCH> target =
                    blaze::trans( blaze::rows( training_data.tdata,
                            [&i](size_t j){ return i*BATCH+j; }, BATCH ) );

            forward( var, xvec, zvec );

            std::get<L::value-1>(deltavec) = loss.template gradient<at<L::value, N...>(),BATCH>(
                        target, std::get<L::value-1>(zvec) );

            backward( var, gradient, xvec, deltavec, zvec );

            objective += loss.template evaluate<at<L::value, N...>(),BATCH>(
                        target, std::get<L::value-1>(zvec) ) / BATCH;

        }

        /**
         * @brief run function; compute backpropagation
         * @param var current position
         * @param info optimisation metainfo which are needed during the iterations
         * @param gradient the computed gradients; the return value
         * @param objective the loss at the current position
         */

        void compute( const variable_t& var, variable_t& gradient, T& objective ) const {
            xdata_t xvec; zdata_t zvec; zdata_t deltavec;

            for( int i = 0; i < training_data.idata.rows() / BATCH ; i++) {
                std::get<0>( xvec ) = blaze::trans( blaze::rows( training_data.idata, [&i](size_t j){ return i*BATCH+j; }, BATCH ) );

                matrix_t<at<L::value, N...>(), BATCH> target =
                        blaze::trans( blaze::rows( training_data.tdata, [&i](size_t j){ return i*BATCH+j; }, BATCH ) );

                forward( var, xvec, zvec );

                std::get<L::value-1>(deltavec) = loss.template gradient<at<L::value, N...>(),BATCH>(
                          target, std::get<L::value-1>(zvec) );

                backward( var, gradient, xvec, deltavec, zvec );

                objective += loss.template evaluate<at<L::value, N...>(),BATCH>(
                            target, std::get<L::value-1>(zvec) ) / BATCH;
            }

        }



        /**
         * @brief forward function; compute forwardpropagation
         * @param layers weights and biases at each layer
         * @param x
         * @param z
         */

        void forward( const variable_t &layers, xdata_t &x, zdata_t &z) const {

            std::for_range<0,L::value-1>([&]<auto I>(){
                auto& layer = std::get<I>(layers);

                std::get<I>(z) =  layer.weight * std::get<I>(x) + blaze::expand<BATCH>(layer.bias);
                std::get<I+1>(x) = ATYPE<T>::template forward<at<I+1,N...>(),BATCH>(
                                        std::get<I>(z) );


            });

            auto& layer = std::get<L::value-1>(layers);
            std::get<L::value-1>(z) = layer.weight *
                    std::get<L::value-1>(x) + blaze::expand<BATCH>( layer.bias );

        }

        /**
         * @brief backward function; compute backpropagation
         * @param layers weights and biases at each layer
         * @param gradient gradient with respect to the weights and biases
         * @param x
         * @param delta gradients with respect to the layer inputs
         * @param z
         */

          void backward( const variable_t &layers, variable_t &gradient,
                         xdata_t &x, zdata_t &delta, zdata_t &z) const {

              std::for_range<0,L::value-1>([&]<auto I>(){

                  auto &grad = std::get<L::value-I-1>(gradient);
                  grad.bias += blaze::reduce<blaze::rowwise>(std::get<L::value-I-1>(delta),
                                               blaze::Add()) / BATCH;
                  grad.weight +=  std::get<L::value-I-1>(delta)
                       *  blaze::trans( std::get<L::value-I-1>(x) ) / BATCH ;

                  auto tmp1 = blaze::trans( std::get<L::value-I-1>(delta) )
                                           * std::get<L::value-I-1>(layers).weight;

                  auto tmp2 = blaze::trans( ATYPE<T>::template
                            derivative<at<L::value-I-2,N...>(),BATCH>(  std::get<L::value-I-2>(z) ));

                  std::get<L::value-I-2>(delta) = blaze::trans( tmp1 % tmp2 );


              });


              auto &grad = std::get<0>(gradient);
              grad.bias += blaze::reduce<blaze::rowwise>(std::get<0>(delta), blaze::Add()) / BATCH;
              grad.weight +=  std::get<0>(delta) *  blaze::trans( std::get<0>(x) ) / BATCH;


          }


    };

}

#endif // __LIPNET_BACKPROPAGATION_HPP__
