dnl
AC_INIT(maketrees, 0.0.1, [Antonio Carzaniga (firstname.lastname@usi.ch)])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE
AM_SILENT_RULES([yes])
AM_CONFIG_HEADER(config.h:config.h.in)
AC_PREFIX_DEFAULT(/opt)
#
AC_PROG_CC
AC_PROG_CXX
AC_LANG(C++)
#
LT_INIT
#
AC_HEADER_TIME
#
AC_CHECK_RDTSC([AC_DEFINE([HAVE_RDTSC], [], [Intel rdtsc instruction])])
AC_CHECK_RDTSCP([AC_DEFINE([HAVE_RDTSCP], [], [Intel rdtscp instruction])])
#
AC_CONFIG_FILES([Makefile])
#
AC_OUTPUT
