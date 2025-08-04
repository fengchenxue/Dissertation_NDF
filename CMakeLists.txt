cmake_minimum_required(VERSION 3.20)
project(Dissertation_NDF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#----python binding------
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(pybind11)

#-----sub project-------
add_subdirectory(external/virtualgonio)

target_include_directories(GonioLib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/external/virtualgonio/src"
)

set(VIRTUALGONIO_TARGETS
    GonioLib
    SurfaceGenLib
    distrib_to_surface
    plot_scattering
    plot_slope_distribution
    tekari_converter
    virtual_gonio
)

foreach(TARGET ${VIRTUALGONIO_TARGETS})
    set_target_properties(${TARGET} PROPERTIES
        FOLDER "Dependencies/VirtualGonio"
    )
endforeach()

#-----main project-----------
add_executable(Dissertation_NDF src/main.cpp)

target_link_libraries(Dissertation_NDF PRIVATE
    GonioLib
    SurfaceGenLib
)

pybind11_add_module(ndf_py src/wrapper.cpp)

target_link_libraries(ndf_py PRIVATE
    GonioLib
    SurfaceGenLib
)

set_target_properties(ndf_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)