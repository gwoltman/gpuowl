# Writes OUT (version.inc) with the version string: PRPLL_VERSION if set, otherwise the basename of git describe
# in SRC as the Makefile does, or "unknown" outside a git checkout.  OUT is only rewritten when the string changes,
# so an unchanged version does not recompile version.cpp.
if (NOT PRPLL_VERSION)
  execute_process(COMMAND git describe --tags --long --dirty --always
                  WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE PRPLL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  get_filename_component(PRPLL_VERSION "${PRPLL_VERSION}" NAME)
endif()
if (NOT PRPLL_VERSION)
  message(WARNING "No git checkout to take the version from, building as \"unknown\"; pass -DPRPLL_VERSION=... to set it")
  set(PRPLL_VERSION unknown)
endif()
file(WRITE ${OUT}.new "\"${PRPLL_VERSION}\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OUT}.new ${OUT})
file(REMOVE ${OUT}.new)
