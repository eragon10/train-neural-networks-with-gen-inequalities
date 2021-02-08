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

#include <lipnet/optimizer.hpp>
#include <lipnet/statistics.hpp>

#include <lipnet/loader/loader.hpp>

#include <lipnet/lipschitz/barrier.hpp>

#include <cereal/types/vector.hpp>
#include <lyra/lyra.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <addon/image.h>

using namespace lipnet;

struct image_t {
    typedef blaze::StaticVector<double,3,blaze::columnVector> pixel_t;
    explicit image_t( size_t w, size_t h )
        : width{ w }, height{ h } { data.resize(w*h); }

    inline pixel_t& operator()(const size_t &x, const size_t &y) {
                return data[y*width+x]; }

    size_t width, height;
    std::vector<pixel_t> data;
};

bool save( const std::string &path, const image_t &img ) {
    size_t size = img.width*img.height*3;
    std::unique_ptr<unsigned char[]> data( new unsigned char[size] );

    for (size_t i = 0; i < img.height*img.width; ++i) {
        auto &pixel = img.data[i];
        //auto &pixel = img.data[ (img.height-i/img.width-1)*img.width + i % img.width];
        for (size_t j = 0; j<3; j++)
            data[i*3+j] = (unsigned char)(255 * std::max(0., std::min(1., pixel[j] )));;
    }

    stbi__flip_vertically_on_write = 1;
    return 0 == stbi_write_png( path.c_str(), img.width, img.height,
                                3, (const void*) data.get(), 3*img.width );
}



auto load_data( const std::string &filename ) {
    network_data_t<double,2,3> data;

    auto opt = loader_t<double>::load(filename);
    if( !opt.has_value() )
        throw std::string{"could not load file"};

    data.idata = blaze::trans( blaze::submatrix( opt.value(), 0, 0,
                           2, opt.value().columns() ));

    auto last = blaze::row( opt.value(), 2);
    data.tdata = blaze::trans( make_one_hot<double>( blaze::trans(last), 3 ) );

    return std::move(data);
}


int main(int argc, char **argv)
{
    std::string modelfile = "model.json";
    std::string surffile = "topology.png";

    size_t nx = 60, ny = 60;

    bool type = false;
    bool show_help = false;
    auto cli
            = lyra::help(show_help)
            | lyra::opt(modelfile, "modelfile")
                  ["-i"]["--input"]("read model as json from 'modelfile'")
            | lyra::opt(surffile, "surffile")
                  ["-o"]["--output"]("save surface topology of network to 'surffile'")
            | lyra::opt(type, "type")
                  ["-t"]["--type"]("image or value file")
            | lyra::opt(nx, "nx")
                  ["-x"]["--numberx"]("resolution in x direction")
            | lyra::opt(ny, "ny")
                  ["-y"]["--numbery"]("resolution in y direction");

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



    typedef blaze::StaticVector<double,2,blaze::columnVector> vector_t;
    typedef blaze::DynamicMatrix<double,blaze::rowMajor> matrix_t;
    network_t<double, tanh_activation_t, 2, 10, 10, 3> network;


    std::ifstream is( modelfile );
    {
        cereal::JSONInputArchive archive(is);
        archive( cereal::make_nvp("model", network) );
    }

    is.close();




    if( type ) {
    std::ofstream stream( surffile );
       csv2::Writer<csv2::delimiter<','>> writer(stream);

    for(int i=0; i < nx+3; i++)
        for(int j=0; j < ny+3; j++)
        {
            double x =  -1.0+2.0/nx*i, y = -1.0+2.0/ny*j;
            auto res = blaze::softmax(network.query( vector_t{x,y} ));

            std::array<std::string,5> row = { std::to_string(x),
                                             std::to_string(y),
                                             std::to_string(res[0]),
                                             std::to_string(res[1]),
                                             std::to_string(res[2]) };
            writer.write_row( row );


            /*blaze::row(inputmatrix, i*ny+j)
                = {x,y, res[0], res[1], res[2]};*/

        }

    stream.close();

    } else {
        image_t img( nx, ny );
        blaze::StaticVector<double,3,blaze::columnVector>
                pixel = {1, 1, 1};
        blaze::StaticVector<double,3,blaze::columnVector>
                 weight = { 1.0, 0.5, 0.0 } ;

        for(int i=0; i < nx; i++)
            for(int j=0; j < ny; j++)
            {
                double x =  -1.0+2.0/nx*i, y = -1.0+2.0/ny*j;

                //auto res = blaze::softmax(network.query( vector_t{x,y} ));
                //std::cout << "res: " << blaze::trans(res);
                //img(i,j) = pixel * blaze::sum( (res * weight) );
                img( i, j ) = blaze::softmax(network.query( vector_t{x,y} ));
            }

        save( surffile, img );
    }


}

