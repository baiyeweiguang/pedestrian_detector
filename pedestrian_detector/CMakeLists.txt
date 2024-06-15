cmake_minimum_required(VERSION 3.10)
project(pedestrian_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror -lSDL2_mixer -lSDL2)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
find_package(SDL2 REQUIRED)
# find_package(SDL2_mixer REQUIRED)
set(SDL2_mixer_LIBRARIES "/home/zcf/Downloads/SDL2_mixer-2.8.0/build/libSDL2_mixer.so")
set(SDL2_mixer_INCLUDE_DIRS "/home/zcf/Downloads/SDL2_mixer-2.8.0/include")
ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${SDL2_mixer_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} openvino::frontend::onnx openvino::runtime ${SDL2_LIBRARIES} ${SDL2_mixer_LIBRARIES})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN pdn::PedestrianDetectorNode
  EXECUTABLE pedestrian_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()
endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
  audio
  config
  launch
)
