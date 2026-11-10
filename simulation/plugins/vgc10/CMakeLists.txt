cmake_minimum_required(VERSION 3.16)
project(Vgc10Vacuum LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs10 REQUIRED)
add_library(Vgc10Vacuum SHARED Vgc10Vacuum.cc)
target_link_libraries(Vgc10Vacuum PRIVATE
    gz-sim8::gz-sim8
    gz-plugin2::gz-plugin2
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10)
