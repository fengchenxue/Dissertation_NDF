cmake_minimum_required(VERSION 3.20)
project(Dissertation_NDF)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(external/virtualgonio)

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


add_executable(Dissertation_NDF src/main.cpp)

target_link_libraries(Dissertation_NDF PRIVATE
    GonioLib
    SurfaceGenLib
)