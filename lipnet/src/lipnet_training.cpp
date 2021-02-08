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


#include <lipnet/extern/nn_lipcalc.hpp>


#include <lipnet/optimizer.hpp>
#include <lipnet/statistics.hpp>

#include <lipnet/loader/loader.hpp>

#include <lipnet/lipschitz/barrier.hpp>

#include <cereal/types/vector.hpp>
#include <lyra/lyra.hpp>

using namespace lipnet;


enum choice_t : size_t {
    NOM = 0,
    L2 = 1,
    PRO_SIMPLE = 2,
    PRO = 3,
    BARR = 4,
    BARRWOT = 5,
    BARRPRE = 6,
    BARRF = 7,
    BARRWOTF = 8,
    BARRPREF = 9
};

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

    double diff = 1e-8;
    double threshold = 1e-8;
    size_t window = 300;

    size_t centralpathsteps = 5;
    double rho = 0.1;
    double beta = 5;

    double rhodec = 0.5;
    double alphadec = 0.5;

    double beta1 = 0.9;
    double beta2 = 0.999;

    double tparam = 100;
    double initweights = 0.1;

    size_t maxiter = 1e5;

    int method = choice_t::NOM;

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
            | lyra::opt(diff, "diff")
                  ["-d"]["--diff"]("stopping criterion (default: 1e-8)")
            | lyra::opt(threshold, "threshold")
                  ["-t"]["--threshold"]("threshold for expo window loss decrease stopping criterion (default: 1e-8)")
            | lyra::opt(window, "window")
                  ["-w"]["--window"]("window for expo window loss decrease stopping criterion (default: 1e-8)")
            | lyra::opt(centralpathsteps, "centralpathsteps")
                  ["-c"]["--steps"]("centralpathsteps (default: 5)")
            | lyra::opt(rho, "rho")
                  ["-r"]["--rho"]("l2-regularisation or log det parameter (default: 0.1)")
            | lyra::opt(rhodec, "rhodec")
                  ["-x"]["--rhodec"]("rhodec (default: 0.5)")
            | lyra::opt(tparam, "tparam")
                  ["-k"]["--tparam"]("tparam fro projection cp (default: 100)")
            | lyra::opt(maxiter, "maxiter")
                  ["-m"]["--maxiter"]("maxm iteration steps (default: 1e5)")
            | lyra::opt(beta, "beta")
                  ["-b"]["--beta"]("decrease parameter for trivial stopping criterion")
            | lyra::opt(beta1, "beta1")
                  ["-q"]["--beta1"]("adam beta1 param")
            | lyra::opt(beta2, "beta2")
                  ["-p"]["--beta2"]("adam beta2 param")
            | lyra::opt(initweights, "initweights")
                  ["-i"]["--initweights"]("initweights variance")
            | lyra::arg( method, "method").help("method to train the network  (default: 5, only barrier) ")
                    .required();

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

    auto data = load_data<INPUTS::value,OUTPUTS::value>( datafile );
    auto nn = nn_t();

    switch ( method ) {
    case choice_t::NOM:{

        typedef network_problem_batch_t<double, tanh_activation_t,
                      cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                    HIDDEN2::value, OUTPUTS::value> pro_nn_t ;
        typedef adam_momentum_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                typename pro_nn_t::variable_t> solver_t;

        solver_t solver( solver_t::parameter_t{ maxiter, diff, 1e-4, alpha, beta1, beta2, 1e-8} );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data)  );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;



        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);


        break; }
    case choice_t::L2: {

        typedef network_problem_batch_l2_t<double, tanh_activation_t,
                      cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                    HIDDEN2::value, OUTPUTS::value> pro_nn_t ;
        typedef adam_momentum_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                typename pro_nn_t::variable_t> solver_t;

        solver_t solver( solver_t::parameter_t{ maxiter, diff, 1e-4, alpha, beta1, beta2, 1e-8} );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data) , rho );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }
    case choice_t::PRO_SIMPLE: {

        typedef network_problem_projection_t<double, tanh_activation_t,
                cross_entropy_t, BATCH::value,INPUTS::value, HIDDEN1::value,
                HIDDEN2::value, OUTPUTS::value> pro_nn_t;

        typedef gradient_descent_projected_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                 typename pro_nn_t::variable_t> solver_t;

        solver_t solver( solver_t::parameter_t{ maxiter, diff, alpha, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz, tparam );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }

    case choice_t::PRO: {
        typedef network_problem_projection_t<double, tanh_activation_t,
             cross_entropy_t, BATCH::value,INPUTS::value, HIDDEN1::value,
             HIDDEN2::value, OUTPUTS::value> pro_nn_t;

            // gradient_descent_projected_t
        typedef adam_projected_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                 typename pro_nn_t::variable_t> solver_t;

        solver_t solver( solver_t::parameter_t{ maxiter, diff, threshold, window, alpha, beta1, beta2, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz, tparam );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }

    case choice_t::BARRWOT: {

        typedef network_problem_log_barrier_wot_t<double, tanh_activation_t,
                      cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

        typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                typename pro_nn_t::variable_t> solver_t;

        typename pro_nn_t::param_t tinit;
        std::get<0>( tinit ) = blaze::uniform( 10, 1e2 );
        std::get<1>( tinit ) = blaze::uniform( 10, 1e2 );

        solver_t solver( solver_t::parameter_t{ maxiter, centralpathsteps,
                                        diff, threshold, window, rho ,alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), std::move(tinit), lipschitz );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }

    case choice_t::BARRPRE: {

        typedef network_problem_batch_t<double, tanh_activation_t,
                 cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                HIDDEN2::value, OUTPUTS::value> ppro_nn_t ;
        typedef adam_momentum_t<double, ppro_nn_t, typename ppro_nn_t::variable_t,
                  typename ppro_nn_t::variable_t> psolver_t;

        auto pdata = load_data<INPUTS::value,OUTPUTS::value>( datafile );
        typename ppro_nn_t::variable_t pinit = generator_t<
                typename ppro_nn_t::variable_t>::make( initweights );

        psolver_t psolver( psolver_t::parameter_t{ maxiter, diff, 1e-4, alpha, beta1, beta2, 1e-8 },
                         [&](const double &fx, const typename ppro_nn_t::variable_t &var,
                            const typename ppro_nn_t::variable_t &grad) -> bool {
                                double L = 1.0; std::for_range<0,3>([&]<auto I>(){
                                   auto &w = std::get<I>(var).weight;
                                   L *= blaze::max( blaze::reduce<blaze::columnwise>(blaze::abs(w), blaze::Add())); });
                                return L < lipschitz;
                         });
        ppro_nn_t pprob ( cross_entropy_t<double>(), std::move(pdata)  );

        psolver_t::main_statistics_t pstats;
        auto [ w, v ] = psolver( pprob, std::move(pinit), pstats );
        auto [ lip, tparam ] = network_libcalc_t<double,INPUTS::value, HIDDEN1::value,
                          HIDDEN2::value, OUTPUTS::value>::solve( w );
        std::cout << "pretrainig finished:  => L: " << lip << "\n";



        typedef network_problem_log_barrier_t<double, tanh_activation_t,
            cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                  HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

        typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                        typename pro_nn_t::variable_t> solver_t;

        solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                    diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights, 0.1 );
        init.W = std::move(w);
        std::get<0>(init.t) = blaze::subvector<0,HIDDEN1::value>(tparam);
        std::get<1>(init.t) = blaze::subvector<HIDDEN1::value,HIDDEN2::value>(tparam);


        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights.W;

        dumptodisk(modelfile, "model", nn);


        std::ofstream oss( statsfile );{
            cereal::JSONOutputArchive archive(oss);
            archive( cereal::make_nvp("prerun", pstats) );
            archive( cereal::make_nvp("run", stats) );
        } oss.close();
        //dumptodisk(statsfile, "run", stats);
        //dumptodisk(statsfile, "prerun", pstats);

        break; }

    case choice_t::BARR: {

        typedef network_problem_log_barrier_t<double, tanh_activation_t,
                      cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

        typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                typename pro_nn_t::variable_t> solver_t;

        solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                    diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );

        typename pro_nn_t::variable_t init = generator_t<
                typename pro_nn_t::variable_t>::make( initweights, 0.1 );
        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights.W;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }

    case choice_t::BARRF: {
        typedef network_problem_log_barrier_t<double, tanh_activation_t,
                   cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                             HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

        typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                 typename pro_nn_t::variable_t, true> solver_t;

        solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                 diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );

        typename pro_nn_t::variable_t init = generator_t<
              typename pro_nn_t::variable_t>::make( initweights, 0.1 );

        typename solver_t::main_statistics_t stats;
        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights.W;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }

    case choice_t::BARRWOTF: {
        typedef network_problem_log_barrier_wot_t<double, tanh_activation_t,
                   cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                       HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

        typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                  typename pro_nn_t::variable_t, true> solver_t;

        typename pro_nn_t::param_t tinit;
        std::get<0>( tinit ) = blaze::uniform( 10, 1e2 );
        std::get<1>( tinit ) = blaze::uniform( 10, 1e2 );

        solver_t solver( solver_t::parameter_t{ maxiter, centralpathsteps,
                      diff, threshold, window, rho ,alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );
        pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), std::move(tinit), lipschitz );

        typename pro_nn_t::variable_t init = generator_t<
        typename pro_nn_t::variable_t>::make( initweights );
        typename solver_t::main_statistics_t stats;

        auto [ weights, value ] = solver( prob, std::move(init), stats );
        nn.layers = weights;

        dumptodisk(modelfile, "model", nn);
        dumptodisk(statsfile, "run", stats);

        break; }


    case choice_t::BARRPREF: {
        typedef network_problem_batch_t<double, tanh_activation_t,
                 cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                      HIDDEN2::value, OUTPUTS::value> ppro_nn_t ;
        typedef adam_momentum_t<double, ppro_nn_t, typename ppro_nn_t::variable_t,
                  typename ppro_nn_t::variable_t> psolver_t;

         auto pdata = load_data<INPUTS::value,OUTPUTS::value>( datafile );
         typename ppro_nn_t::variable_t pinit = generator_t<
         typename ppro_nn_t::variable_t>::make( initweights );

         psolver_t psolver( psolver_t::parameter_t{ maxiter, diff, 1e-4, alpha, beta1, beta2, 1e-8 },
                 [&](const double &fx, const typename ppro_nn_t::variable_t &var,
                     const typename ppro_nn_t::variable_t &grad) -> bool {
                          double L = 1.0; std::for_range<0,3>([&]<auto I>(){
                          auto &w = std::get<I>(var).weight;
                          L *= blaze::max( blaze::reduce<blaze::columnwise>(blaze::abs(w), blaze::Add())); });
                          return L < lipschitz;
                                 });

         ppro_nn_t pprob ( cross_entropy_t<double>(), std::move(pdata)  );

         psolver_t::main_statistics_t pstats;
         auto [ w, v ] = psolver( pprob, std::move(pinit), pstats );
         auto [ lip, tparam ] = network_libcalc_t<double,INPUTS::value, HIDDEN1::value,
                   HIDDEN2::value, OUTPUTS::value>::solve( w );

         std::cout << "pretrainig finished:  => L: " << lip << "\n";


         typedef network_problem_log_barrier_t<double, tanh_activation_t,
              cross_entropy_t, BATCH::value, INPUTS::value, HIDDEN1::value,
                       HIDDEN2::value, OUTPUTS::value> pro_nn_t ;

         typedef adam_barrier_t<double, pro_nn_t, typename pro_nn_t::variable_t,
                    typename pro_nn_t::variable_t,true> solver_t;

         solver_t solver(  solver_t::parameter_t{ maxiter, centralpathsteps,
                      diff, threshold, window, rho, alpha, beta1, beta2, alphadec, rhodec, 1e-8 } );

         pro_nn_t prob ( cross_entropy_t<double>(), std::move(data), lipschitz );

         typename pro_nn_t::variable_t init = generator_t<
         typename pro_nn_t::variable_t>::make( initweights, 0.1 );
         init.W = std::move(w);

         std::get<0>(init.t) = blaze::subvector<0,HIDDEN1::value>(tparam);
         std::get<1>(init.t) = blaze::subvector<HIDDEN1::value,HIDDEN2::value>(tparam);

         typename solver_t::main_statistics_t stats;
         auto [ weights, value ] = solver( prob, std::move(init), stats );
         nn.layers = weights.W;

         dumptodisk(modelfile, "model", nn);

         std::ofstream oss( statsfile );{
             cereal::JSONOutputArchive archive(oss);
             archive( cereal::make_nvp("prerun", pstats) );
             archive( cereal::make_nvp("run", stats) );
         } oss.close();
                //dumptodisk(statsfile, "run", stats);
                //dumptodisk(statsfile, "prerun", pstats);
         break; }


    }


}
