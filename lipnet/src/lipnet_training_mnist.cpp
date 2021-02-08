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


#include <lipnet/problem/nn_problem_batch.hpp>
#include <lipnet/problem/nn_problem_batch_l2.hpp>

#include <lipnet/problem/nn_problem_batch_admm.hpp>

#include <lipnet/problem/nn_problem_liptrain_barrier.hpp>
#include <lipnet/problem/nn_problem_liptrain_barrier_wot.hpp>
#include <lipnet/problem/nn_problem_liptrain_projection.hpp>


#include <lipnet/optimizer.hpp>
#include <lipnet/statistics.hpp>

#include <lipnet/loader/loader.hpp>
#include <lipnet/loader/container.hpp>

#include <lipnet/lipschitz/barrier.hpp>

#include <cereal/types/vector.hpp>
#include <lyra/lyra.hpp>

using namespace lipnet;


template<typename NN>
int dumptodisk( const std::string &path, const std::string &name, NN &nn ) {
    std::ofstream oss( path );
    {
        cereal::JSONOutputArchive archive(oss);
        archive( cereal::make_nvp(name, nn) );
    }
    oss.close();
    return 0;
}


enum choice_t : size_t {
    NOM = 0,
    BARR = 1
};


int main(int argc, char **argv)
{
    typedef std::integral_constant<size_t,196>  INPUTS;
    typedef std::integral_constant<size_t,100> HIDDEN1;
    typedef std::integral_constant<size_t,40> HIDDEN2;
    typedef std::integral_constant<size_t,10>  OUTPUTS;

    typedef std::integral_constant<size_t, 60000> BATCH;

    std::string datafile = "mnist_training.json";
    std::string modelfile = "model_mnist.json";
    std::string statsfile = "stats_mnist.json";

    double lipschitz = 20;
    double alpha = 0.001;
    double diff = 1e-8;
    size_t centralpathsteps = 3;
    size_t maxiter = 1e4;
    double threshold = 1e-8;
    size_t window = 300;

    double rhodec = 0.5;
    double alphadec = 0.5;

    double rho = 0.1;

    double beta1 = 0.9, beta2 = 0.999;

    int method = choice_t::NOM;
    bool feasbility_enabled = true;

    bool show_help = false;
    auto cli
            = lyra::help(show_help)
            | lyra::opt(datafile, "inputfile")
                  ["-f"]["--file"]("read datapoints as csv from 'inputfile'")
            | lyra::opt(modelfile, "modelfile")
                  ["-o"]["--output"]("save model as json to 'modelfile'")
            | lyra::opt(statsfile, "statsfile")
                  ["-s"]["--stats"]("save statistics about optimization to 'statsfile'")
            | lyra::opt(lipschitz, "lipschitz")
                  ["-l"]["--lipschitz"]("set enforceing lipschitz constant")
            | lyra::opt(alpha, "alpha")
                  ["-a"]["--alpha"]("set stepsize alpha (default: 0.02)")
            | lyra::opt(alphadec, "alphadec")
                  ["-y"]["--alphadec"]("alphadec (default: 0.5)")
            | lyra::opt(threshold, "threshold")
                  ["-t"]["--threshold"]("threshold for expo window loss decrease stopping criterion (default: 1e-8)")
            | lyra::opt(window, "window")
                  ["-w"]["--window"]("window for expo window loss decrease stopping criterion (default: 1e-8)")
            | lyra::opt(beta1, "beta1")
                  ["-q"]["--beta1"]("adam beta1 param")
            | lyra::opt(beta2, "beta2")
                  ["-p"]["--beta2"]("adam beta2 param")
            | lyra::opt(diff, "diff")
                  ["-d"]["--diff"]("stopping criterion (default: 1e-8)")
            | lyra::opt(centralpathsteps, "centralpathsteps")
                  ["-c"]["--steps"]("centralpathsteps (default: 5)")
            | lyra::opt(rho, "rho")["-r"]["--rho"]("asdasd")
            | lyra::opt(rhodec, "rhodec")
                  ["-x"]["--rhodec"]("rhodec (default: 0.5)")
            | lyra::opt(feasbility_enabled, "feasbility_enabled")["-e"]["--fenabled"]("default true")
            | lyra::opt(maxiter, "maxiter")["-m"]["--maxiter"]("maxiter")
            | lyra::arg(method, "method").help("method to train the network").required();

    auto result = cli.parse({ argc, argv });
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        std::cerr << cli << "\n";  return 1;
    }

    if (show_help) {
        std::cout << cli << "\n";  return 0;
    }

     data_container_t<double> mnist;
     std::ifstream iss( datafile );
     {
         cereal::JSONInputArchive archive( iss );
         archive( cereal::make_nvp("mnist", mnist) );
     }

     iss.close();

     std::cout << "data loaded..."  << "\n";
     network_data_t<double,INPUTS::value,OUTPUTS::value> data{
         std::move( mnist.x ), std::move( mnist.y )
     };


     typedef network_t<double, tanh_activation_t, INPUTS::value, HIDDEN1::value,
             HIDDEN2::value, OUTPUTS::value> nn_t;
     auto nn = nn_t();


     switch (method) {
     case choice_t::BARR: {

         typedef network_problem_log_barrier_t<double, tanh_activation_t,
                         cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                 HIDDEN2::value, OUTPUTS::value> pro_nn_t ;
         typename pro_nn_t::variable_t init
             = generator_t<typename pro_nn_t::variable_t>::make( 0.1, 2.0 );
         if (feasbility_enabled) {
             typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                     typename pro_nn_t::variable_t, true> solver_t;
             //solver_t solver( solver_t::parameter_t{ (size_t) 100, 1e-6, 0.5, 0.03, 0.9, 0.999, 1e-8 } );
             solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                          diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec,  1e-8 } );
             pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );
             solver_t::main_statistics_t stats;
             auto [ weights, value ] = solver( prob, std::move(init), stats );
             nn.layers = weights.W;

             dumptodisk( statsfile, "run", stats );
         } else {
             typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                     typename pro_nn_t::variable_t, false> solver_t;
             //solver_t solver( solver_t::parameter_t{ (size_t) 100, 1e-6, 0.5, 0.03, 0.9, 0.999, 1e-8 } );
             solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                          diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec,  1e-8 } );
             pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );
             solver_t::main_statistics_t stats;
             auto [ weights, value ] = solver( prob, std::move(init), stats );
             nn.layers = weights.W;


             dumptodisk( statsfile, "run", stats );
         }

         break;
     }
     case choice_t::NOM: {

        typedef network_problem_batch_t<double, tanh_activation_t,
                 cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                         HIDDEN2::value, OUTPUTS::value> pro_nn_t ;
        typename pro_nn_t::variable_t init = generator_t<typename pro_nn_t::variable_t>::make( 0.1 );

        typedef adam_momentum_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                   typename pro_nn_t::variable_t> solver_t;

        solver_t solver(  solver_t::parameter_t{ maxiter, diff, 1e-12, alpha, beta1, beta2, 1e-8 } );

        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data) );

        solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;
        dumptodisk( statsfile, "run", stats );

        break;
     }
     }

     dumptodisk( modelfile, "model", nn );



}
