cmake_minimum_required(VERSION 3.0)


FILE(GLOB_RECURSE LIPNET_SOURCES "src/*.cpp")
FILE(GLOB_RECURSE LIPNET_HEADERS "include/*.hpp")
FILE(GLOB_RECURSE LIPNET_PRIVATE "src/*.hpp")

add_definitions(-Wall -Wshadow -Woverloaded-virtual -ansi -pedantic -O3 -mavx -mfma -fopenmp -DNDEBUG -DMTL_HAS_BLAS -msse2 -mfpmath=sse -DBLAZE_USE_CPP_THREADS  )
    #-msse2 -mfpmath=sse -ftree-vectorizer-verbose=5)
    
    
    

add_library(lipnet INTERFACE)
target_include_directories(lipnet
    INTERFACE include)
target_link_libraries(lipnet INTERFACE
    blaze csv2 cereal lyra mosek)
    
    
    

add_executable(lipnet_training
    src/lipnet_training.cpp
    ${LIPNET_HEADERS}
    ${LIPNET_PRIVATE})
set_property(TARGET lipnet_training PROPERTY CXX_STANDARD 17)
target_link_libraries(lipnet_training lipnet)





add_executable(lipnet_training_mnist
    src/lipnet_training_mnist.cpp)
set_property(TARGET lipnet_training_mnist PROPERTY CXX_STANDARD 17)
target_link_libraries(lipnet_training_mnist lipnet)



add_executable(lipnet_admm_training
    src/lipnet_admm_training.cpp)
set_property(TARGET lipnet_admm_training PROPERTY CXX_STANDARD 17)
target_link_libraries(lipnet_admm_training lipnet)





add_executable(plotting_objectivsurface
    src/plotting_objectivsurface.cpp
    ${LIPNET_HEADERS}
    ${LIPNET_PRIVATE})
set_property(TARGET plotting_objectivsurface PROPERTY CXX_STANDARD 17)
target_include_directories(plotting_objectivsurface PUBLIC include PRIVATE src)
target_link_libraries(plotting_objectivsurface blaze csv2 cereal lyra)


add_executable(plotting_statistics
    src/plotting_statistics.cpp
    ${LIPNET_HEADERS}
    ${LIPNET_PRIVATE})
set_property(TARGET plotting_statistics PROPERTY CXX_STANDARD 17)
target_include_directories(plotting_statistics PUBLIC include PRIVATE src)
target_link_libraries(plotting_statistics blaze csv2 cereal lyra mosek)






