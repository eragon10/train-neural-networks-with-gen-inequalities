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

#include <lipnet/problem/nn_problem_batch.hpp>
#include <lipnet/problem/nn_problem_batch_admm.hpp>
#include <lipnet/problem/nn_problem_liptrain_admm.hpp>

#include <lipnet/optimizer.hpp>
#include <lipnet/statistics.hpp>


#include <lipnet/loader/loader.hpp>


#include <lipnet/extern/nn_lipcalc.hpp>
#include <lipnet/extern/nn_liptrain_enforcing.hpp>

#include <lipnet/lipschitz/barrier.hpp>
#include <lipnet/statistics.hpp>

#include <cereal/types/vector.hpp>
#include <lyra/lyra.hpp>



using namespace lipnet;



template<size_t I, size_t O>
auto load_data( const std::string &filename ) {
    network_data_t<double,I,O> data;

    auto opt = loader_t<double>::load(filename);
    if( !opt.has_value() )
        throw std::string{"could not load file"};

    data.idata = blaze::trans( blaze::submatrix( opt.value(), 0, 0,
                           I, opt.value().columns() ));

    auto last = blaze::row( opt.value(), I );
    data.tdata = blaze::trans( make_one_hot<double>( blaze::trans(last), O ) );

    return std::move(data);
}


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

int main(int argc, char **argv)
{
    typedef std::integral_constant<size_t,2>  INPUTS;
    typedef std::integral_constant<size_t,10> HIDDEN1;
    typedef std::integral_constant<size_t,10> HIDDEN2;
    typedef std::integral_constant<size_t,3>  OUTPUTS;

    typedef std::integral_constant<size_t, 400> BATCH;

    std::string datafile = "data.csv";
    std::string modelfile = "model.json";
    std::string statsfile = "stats.json";

    double lipschitz = 50;
    double alpha = 0.02;

    double rho = 2;
    double diff = 1e-2;

    double beta1 = 0.9;
    double beta2 = 0.999;

    double initweights = 0.1;
    size_t maxiter = 50;


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
            | lyra::opt(rho, "rho")
                  ["-r"]["--rho"]("rho parameter for augmented lagrangian (default: 2)")
            | lyra::opt(diff, "diff")
                  ["-d"]["--diff"]("stopping criterion (default: 1e-2)")
            | lyra::opt(maxiter, "maxiter")
                  ["-m"]["--maxiter"]("maxm iteration steps (default: 50)")
            | lyra::opt(beta1, "beta1")
                  ["-q"]["--beta1"]("adam beta1 param")
            | lyra::opt(beta2, "beta2")
                  ["-p"]["--beta2"]("adam beta2 param")
            | lyra::opt(initweights, "initweights")
                  ["-i"]["--initweights"]("initweights variance");

    auto result = cli.parse({ argc, argv });
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        std::cerr << cli << "\n";  return 1;
    }

    if (show_help) {
        std::cout << cli << "\n";  return 0;
    }


    typedef network_t<double, tanh_activation_t, INPUTS::value, HIDDEN1::value,
                HIDDEN2::value, OUTPUTS::value> nn_t;

    typedef network_problem_batch_t<double, tanh_activation_t,
                  cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
            HIDDEN2::value, OUTPUTS::value> pre_pro_nn_t ;

    typedef adam_momentum_t<double, pre_pro_nn_t, typename pre_pro_nn_t::variable_t,
            typename pre_pro_nn_t::variable_t> pre_solver_t;


    auto pdata = load_data<INPUTS::value,OUTPUTS::value>( datafile );
    typename pre_pro_nn_t::variable_t init = generator_t<
            typename pre_pro_nn_t::variable_t>::make( initweights );

    pre_solver_t psolver;
    pre_pro_nn_t pprob ( cross_entropy_t<double>(), std::move(pdata)  );

    pre_solver_t::main_statistics_t ppstats;
    auto [ weights, pvalue ] = psolver( pprob, std::move(init), ppstats );



   typedef network_t<double, tanh_activation_t, INPUTS::value, HIDDEN1::value,
            HIDDEN2::value, OUTPUTS::value> nn_t;

   typedef network_problem_liptrain_enforcing_adam_t<double, tanh_activation_t,
                 cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
            HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

   typedef admm_optimizer_t<double, pro_nn_t, typename pro_nn_t::variable_t,
           typename pro_nn_t::variable_t, typename pro_nn_t::variable_t> solver_t;


   auto data = load_data<INPUTS::value,OUTPUTS::value>( datafile );
   typename pro_nn_t::variable_t init1 = weights;
   typename pro_nn_t::variable_t init2 = weights;

   solver_t solver( solver_t::parameter_t{ (size_t) maxiter, rho, diff}  );
   pro_nn_t prob ( std::move(data), lipschitz );

   solver_t::main_statistics_t stats;
   auto [ w1, w2, value ] = solver( prob, std::move(init1),
           std::move(init2), stats );
   auto nn = nn_t(); nn.layers = w1;



    dumptodisk(modelfile, "model", nn);
    dumptodisk(statsfile, "run", stats);




}
