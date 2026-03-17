# Findre2.cmake - Find re2 library using pkg-config as fallback
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(RE2 QUIET re2)
endif()

if(NOT RE2_FOUND)
  find_path(RE2_INCLUDE_DIRS re2/re2.h)
  find_library(RE2_LIBRARIES NAMES re2)
  if(RE2_INCLUDE_DIRS AND RE2_LIBRARIES)
    set(RE2_FOUND TRUE)
  endif()
endif()

if(RE2_FOUND AND NOT TARGET re2::re2)
  add_library(re2::re2 IMPORTED INTERFACE)
  set_target_properties(re2::re2 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${RE2_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${RE2_LIBRARIES}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(re2 DEFAULT_MSG RE2_LIBRARIES RE2_INCLUDE_DIRS)
