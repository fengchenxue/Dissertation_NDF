cmake_minimum_required(VERSION 3.20)
project(Dissertation_NDF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
  add_compile_options(/utf-8)
endif()


#----python-------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE_PATH "${Python3_EXECUTABLE}")

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

python3_add_library(ndf_py MODULE src/wrapper.cpp)

target_link_libraries(ndf_py PRIVATE
    pybind11::module
    Python3::Module       
    Python3::Python        
    GonioLib
    SurfaceGenLib
)


set_target_properties(ndf_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ---- copy .pyd to Python script directory
add_custom_command(TARGET ndf_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:ndf_py>"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Python"
)


# ---- generate .pyproj on Windows
if(WIN32)
    file(GLOB_RECURSE PYTHON_SCRIPTS RELATIVE ${CMAKE_SOURCE_DIR} src/python/*.py)
    set(PYTHON_SOURCE_ITEMS "")
    foreach(PYFILE ${PYTHON_SCRIPTS})
        set(PYTHON_SOURCE_ITEMS "${PYTHON_SOURCE_ITEMS}    <Compile Include=\"${PYFILE}\" />\n")
    endforeach()

    set(PYPROJ_OUTPUT "${CMAKE_SOURCE_DIR}/PythonTraining.pyproj")

    configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/python_project.pyproj.in
        ${PYPROJ_OUTPUT}
        @ONLY
    )

    message(STATUS " Generated Visual Studio Python project: ${PYPROJ_OUTPUT}")
endif()