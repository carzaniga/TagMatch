dnl
AC_INIT(maketrees, 0.0.1, [Antonio Carzaniga (firstname.lastname@usi.ch)])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE
AM_SILENT_RULES([yes])
AM_CONFIG_HEADER(config.h:config.h.in)
AC_PREFIX_DEFAULT(/opt)
#
AC_PROG_CXX
AC_LANG(C++)
#
LT_INIT
#
AX_CXX_COMPILE_STDCXX_0X
if test "$ax_cv_cxx_compile_cxx0x_cxx" = yes; then
  CXXFLAGS="-std=c++0x $CXXFLAGS"
elif test "$ax_cv_cxx_compile_cxx0x_gxx" = yes; then
  CXXFLAGS="-std=gnu++0x $CXXFLAGS"
fi
AX_CHECK_COMPILE_FLAG([-Wall], [
  CFLAGS="-Wall $CFLAGS"
  CXXFLAGS="-Wall $CXXFLAGS"
])
AC_OPT_DEBUGGING
AC_OPT_PROFILING
AC_OPT_TIMERS
#
# Boost unit testing framework
#
AX_BOOST_BASE([1.40], [
  AX_BOOST_UNIT_TEST_FRAMEWORK
  if test "x$ax_cv_boost_unit_test_framework" = "xyes"; then
      USE_BOOST_UTF=yes
  else
      USE_BOOST_UTF=no
  fi],  [USE_BOOST_UTF=no])
if test "x$USE_BOOST_UTF" != "xyes"; then
   AC_MSG_WARN([Without the Boost Unit Test Framework you will not be able to run the tests.])
fi
AM_CONDITIONAL([USE_BOOST_UNIT_TEST_FRAMEWORK], [test "x$USE_BOOST_UTF" = "xyes"])
#
AC_CHECK_BUILTIN_CLZL
AC_CHECK_BUILTIN_CTZL
AC_CHECK_BUILTIN_POPCOUNT
#
AC_CONFIG_FILES([Makefile 
		 test/Makefile])
#
AC_OUTPUT
