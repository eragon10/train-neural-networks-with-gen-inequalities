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


#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <lipnet/network/activation.hpp>
#include <lipnet/network/loss.hpp>
#include <lipnet/network/topology.hpp>
#include <lipnet/network/backpropagation.hpp>

#include <lipnet/lipschitz/trivial.hpp>

#include <lipnet/extern/nn_lipcalc.hpp>

#include <lipnet/optimizer.hpp>
#include <lipnet/statistics.hpp>

#include <lipnet/loader/loader.hpp>

#include <lipnet/lipschitz/barrier.hpp>

#include <cereal/types/vector.hpp>
#include <lyra/lyra.hpp>

using namespace lipnet;





auto load_data( const std::string &filename ) {
    network_data_t<double,2,3> data;

    auto opt = loader_t<double>::load(filename);
    if( !opt.has_value() )
        throw std::string{"could not load file"};

    data.idata = blaze::trans( blaze::submatrix( opt.value(), 0, 0,
                           2, opt.value().columns() ));

    auto last = blaze::row( opt.value(), 2 );
    data.tdata = blaze::trans( make_one_hot<double>( blaze::trans(last), 3 ) );

    return std::move(data);
}


int main(int argc, char **argv)
{
    std::string modelfile = "model.json";
    std::string testdatafile = "data.csv";

    size_t nx = 40, ny = 40;

    bool show_help = false;
    auto cli
            = lyra::help(show_help)
            | lyra::opt(modelfile, "modelfile")
                  ["-i"]["--input"]("read model as json from 'modelfile'")
            | lyra::opt(testdatafile, "testdatafile")
                  ["-t"]["--testdata"]("read test dataset from 'testdatafile'");

    auto result = cli.parse({ argc, argv });

    if (!result)
    {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        std::cerr << cli << "\n"; // <1>
        return 1;
    }

    if (show_help)
    {
        std::cout << cli << "\n";
        return 0;
    }



    auto data = load_data( testdatafile );

    typedef blaze::DynamicVector<int,blaze::columnVector> vector_t;
    typedef blaze::StaticVector<double,2,blaze::columnVector> vec_t;
    network_t<double, tanh_activation_t, 2, 10, 10, 3> network;


    std::ifstream is( modelfile );
    {
        cereal::JSONInputArchive archive(is);
        archive( cereal::make_nvp("model", network) );
    }

    is.close();


    //std::cout << data.idata << "\n";
    vector_t list(  data.idata.rows() );

    for(int i=0; i < data.idata.rows(); i++) {

        vec_t in =  blaze::trans(blaze::row(data.idata,i));
        auto obj = network.query( in );

        list[i] = blaze::argmax(blaze::row(data.tdata,i))
                        == blaze::argmax( obj ) ? 1 : 0;
    }

    double acc = ((double) blaze::sum( list )) / list.size();
    auto [ lip, tparam ] = network_libcalc_t<double, 2,10,10,3>::solve( network.layers );


    auto trivlip = calculate_lipschitz_t<double,2,10,10,3>::trivial_lipschitz( network.layers );

    std::cout << "acc: " << acc << "\tlip: " << lip << "\ttrivlip: " << trivlip << "\n";
}

