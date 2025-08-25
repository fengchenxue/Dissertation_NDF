cmake_minimum_required(VERSION 3.23)
project(Dissertation_NDF LANGUAGES CXX)

# Global C++ config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Paths
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(PY_OUT  ${SRC_DIR}/Python)

# ---------------- Python & pybind11 ----------------
# Find Python. Some Windows installs don't export Python3::Module;
# make it optional to avoid hard failure.
find_package(Python3 REQUIRED
  COMPONENTS Interpreter Development
  OPTIONAL_COMPONENTS Development.Module
)

# Find or fetch pybind11 (v3.0 is fine; legacy helper kept for fallback)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.0)
  FetchContent_MakeAvailable(pybind11)
endif()

# --------------- External author library ---------------
# Make sure this subdir defines the targets you link (e.g., GonioLib/SurfaceGenLib)
add_subdirectory(external/virtualgonio)
target_include_directories(GonioLib PUBLIC
  ${CMAKE_SOURCE_DIR}/external/virtualgonio/src)

# --------------- Python extension module ---------------
# Prefer the "new style" for pybind11 >= 3.0 (MODULE + headers);
# fall back to pybind11_add_module if needed.
if(DEFINED pybind11_VERSION AND NOT pybind11_VERSION VERSION_LESS 3.0.0)
  add_library(ndf_py MODULE
    ${SRC_DIR}/wrapper.cpp
  )
  target_link_libraries(ndf_py PRIVATE
    pybind11::headers
    GonioLib
    SurfaceGenLib
  )
  # Link Python3::Module only if CMake exported it (may be missing on some 3.13 installs)
  if(TARGET Python3::Module)
    target_link_libraries(ndf_py PRIVATE Python3::Module)
  endif()
  if(WIN32)
    # Ensure .pyd suffix on Windows
    set_target_properties(ndf_py PROPERTIES SUFFIX ".pyd")
  endif()
else()
  # Legacy helper for older pybind11
  pybind11_add_module(ndf_py
    ${SRC_DIR}/wrapper.cpp
  )
  target_link_libraries(ndf_py PRIVATE
    GonioLib
    SurfaceGenLib
  )
endif()

# ---- MSVC options (AFTER target is created) ----
if(MSVC)
  target_compile_options(ndf_py PRIVATE /utf-8 /Zc:__cplusplus /EHsc)
  set_target_properties(ndf_py PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"   # /MD runtime
  )
endif()

# Output .pyd to src/Python (match your repo)
set_target_properties(ndf_py PROPERTIES
  PREFIX ""                       # ndf_py.pyd
  OUTPUT_NAME "ndf_py"
  RUNTIME_OUTPUT_DIRECTORY "${PY_OUT}"
  LIBRARY_OUTPUT_DIRECTORY "${PY_OUT}"
  ARCHIVE_OUTPUT_DIRECTORY "${PY_OUT}"
)

# Copy runtime DLLs next to the module (so Python can import without PATH hacks)
if(MSVC)
  add_custom_command(TARGET ndf_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:ndf_py> $<TARGET_FILE_DIR:ndf_py>
    COMMAND_EXPAND_LISTS)
endif()

# (Optional) your native exe
add_executable(Dissertation_NDF ${SRC_DIR}/main.cpp)
target_link_libraries(Dissertation_NDF PRIVATE GonioLib SurfaceGenLib)

# (Optional) fix case when generating .pyproj
if(WIN32)
  file(GLOB_RECURSE PYTHON_SCRIPTS RELATIVE ${CMAKE_SOURCE_DIR} src/Python/*.py)
  # ... your pyproj generation here if you keep it ...
endif()
