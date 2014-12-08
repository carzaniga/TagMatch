dnl
dnl AC_DISABLE_COMPILER_OPTIMIZATION
dnl
AC_DEFUN([AC_DISABLE_COMPILER_OPTIMIZATION], [[
  CFLAGS=`echo "$CFLAGS" | sed "s/-O[^ ]*/-O0/g"`
  CXXFLAGS=`echo "$CXXFLAGS" | sed "s/-O[^ ]*/-O0/g"`
]])
dnl
dnl AC_OPT_PROFILING
dnl
AC_DEFUN([AC_OPT_PROFILING], [
AC_ARG_ENABLE(profiling, 
  AC_HELP_STRING([--enable-profiling],
	[include profiling information. Values are "yes", "coverage" and "no" (default is "no")]),
dnl this is to optionally compile with profiling
dnl I don't know too much about this, but it looks like
dnl -pg only works with static libraries, so I'm going to 
dnl disable shared libraries here.
  [ case "$enableval" in
        coverage )
	    CFLAGS_prof='-fprofile-arcs -ftest-coverage'
	    CXXFLAGS_prof='-fprofile-arcs -ftest-coverage'
	    LDFLAGS_prof=''
	    LIBS_prof=''
	    AC_MSG_RESULT([enabling profiling for coverage information])
	    AC_DISABLE_COMPILER_OPTIMIZATION
	    ;;
        * )
	    CFLAGS_prof='-pg'
	    CXXFLAGS_prof='-pg'
	    LDFLAGS_prof='-pg'
	    LIBS_prof=''
	    AC_MSG_RESULT([enabling profiling for performance])
	    ;;
    esac
    AC_DISABLE_SHARED ], 
  [ CFLAGS_prof=''
    CXXFLAGS_prof=''
    LDFLAGS_prof=''
    LIBS_prof=''
    AC_ENABLE_SHARED ])
AC_SUBST(CFLAGS_prof)
AC_SUBST(CXXFLAGS_prof)
AC_SUBST(LDFLAGS_prof)
AC_SUBST(LIBS_prof)
])
dnl
dnl AC_OPT_DEBUGGING
dnl
AC_DEFUN([AC_OPT_DEBUGGING], [
AC_ARG_ENABLE(debugging, 
  AC_HELP_STRING([--enable-debugging],
	[disable compiler optimizations for better debugging (default is NO)]), 
  [[ 
    CFLAGS=`echo "$CFLAGS" | sed "s/-O[^ ]*/-Og/g"`
    CXXFLAGS=`echo "$CXXFLAGS" | sed "s/-O[^ ]*/-Og/g"`
  ]])])
dnl
dnl AC_OPT_FRONTEND_STATS
dnl
AC_DEFUN([AC_OPT_FRONTEND_STATS], [
AC_ARG_ENABLE(frontend-statistics, 
  AC_HELP_STRING([--enable-frontend-statistics],
	[enable front-end statistics (default is NO)]), 
  [ AC_DEFINE(WITH_FRONTEND_STATISTICS, 1, [Enable front-end statistics])])
])
dnl
dnl AC_OPT_DISABLE_BACKEND
dnl
AC_DEFUN([AC_OPT_DISABLE_BACKEND], [
AC_ARG_ENABLE(backend, 
  AC_HELP_STRING([--disable-backend],
	[disable back-end matching (default is ENABLED).  This is for testing purposes, to test the front-end in isolation]), 
  [ AC_DEFINE(BACK_END_IS_VOID, 1, [Disable back-end matching])])
AC_ARG_ENABLE(backend-no-output, 
  AC_HELP_STRING([--disable-backend-no-output],
	[disable back-end matching (default is ENABLED).  This is for testing purposes, to test the front-end in isolation.  This option differs from --disable-backend in that it also does not go through the loop that sets the output.]), 
  [ AC_DEFINE(BACK_END_IS_VOID, 1, [Disable back-end matching])]
  [ AC_DEFINE(NO_FRONTEND_OUTPUT_LOOP, 1, [Disable loop over batch to set output after backend processing])])
])
dnl
dnl AC_OPT_ASSERTIONS
dnl
AC_DEFUN([AC_OPT_ASSERTIONS], [
AC_ARG_ENABLE(assertions, 
  AC_HELP_STRING([--enable-assertions],
	[enable evaluation of assertions (default is NO)]), ,
  [ AC_DEFINE(NDEBUG, 1, [Disables assertions and other debugging code])])
])
dnl
dnl AC_CHECK_BUILTIN_CLZL([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_BUILTIN_CLZL], [
AC_CACHE_CHECK([for __builtin_clzl], [ac_cv_clzl], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
int clzl(unsigned long d) {
    return __builtin_clzl(d);
}
]])], [ ac_cv_clzl=yes ], [ ac_cv_clzl=no ])])
case "$ac_cv_clzl" in
    yes)
        AC_DEFINE(HAVE_BUILTIN_CLZL, 1, [We may use the compiler's built-in clzl function])
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_BUILTIN_CTZL([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_BUILTIN_CTZL], [
AC_CACHE_CHECK([for __builtin_ctzl], [ac_cv_ctzl], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
int ctzl(unsigned long d) {
    return __builtin_ctzl(d);
}
]])], [ ac_cv_ctzl=yes ], [ ac_cv_ctzl=no ])])
case "$ac_cv_ctzl" in
    yes)
        AC_DEFINE(HAVE_BUILTIN_CTZL, 1, [We may use the compiler's built-in ctzl function])
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])

dnl
dnl AC_CHECK_BUILTIN_CAS([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_BUILTIN_CAS], [
AC_CACHE_CHECK([for __sync_bool_compare_and_swap], [ac_cv_cas], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
volatile unsigned int v;
bool f() {
    unsigned int cv;
    return __sync_bool_compare_and_swap(&v, cv, cv + 1);
}
]])], [ ac_cv_cas=yes ], [ ac_cv_cas=no ])])
case "$ac_cv_cas" in
    yes)
        AC_DEFINE(HAVE_BUILTIN_CAS, 1, [We may use the compiler's built-in compare-and-swap function])
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])

dnl
dnl AC_CHECK_BUILTIN_POPCOUNT([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_BUILTIN_POPCOUNT], [
AC_CACHE_CHECK([for __builtin_popcount], [ac_cv_popcount], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
int popcount(unsigned long d) {
    return __builtin_popcount(d);
}
]])], [ ac_cv_popcount=yes ], [ ac_cv_popcount=no ])])
case "$ac_cv_popcount" in
    yes)
        AC_DEFINE(HAVE_BUILTIN_POPCOUNT, 1, [We may use the compiler's built-in popcount function])
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_RDTSC([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_RDTSC], [
AC_CACHE_CHECK([for the rdtsc instruction], [ac_cv_rdtsc], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
long long int rdtsc() {
    long long int x;
    __asm__ volatile ("rdtsc" : "=A" (x));
    return x;
}
]])], [ ac_cv_rdtsc=yes ], [ ac_cv_rdtsc=no ])])
case "$ac_cv_rdtsc" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_RDTSCP([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_RDTSCP], [
AC_CACHE_CHECK([for the rdtscp instruction], [ac_cv_rdtscp], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
long long int rdtscp() {
    long long int x;
    __asm__ volatile ("rdtscp" : "=A" (x));
    return x;
}
]])], [ ac_cv_rdtscp=yes ], [ ac_cv_rdtscp=no ])])
case "$ac_cv_rdtscp" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_MFTB([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_MFTB], [
AC_CACHE_CHECK([for the mftb instruction], [ac_cv_mftb], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
unsigned long get_cycles(void) {
    unsigned long ret;

    __asm__ __volatile__("mftb %0" : "=r" (ret) : );
    return ret;
}
]])], [ ac_cv_mftb=yes ], [ ac_cv_mftb=no ])])
case "$ac_cv_mftb" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_CHECK_MONOTONIC_RAW([action-if-available [, action-if-not-available])
dnl
AC_DEFUN([AC_CHECK_MONOTONIC_RAW], [
AC_CACHE_CHECK([for CLOCK_MONOTONIC_RAW], [ac_cv_monotonic_raw], [
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <time.h>
void f() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);
}
]])], [ ac_cv_monotonic_raw=yes ], [ ac_cv_monotonic_raw=no ])])
case "$ac_cv_monotonic_raw" in
    yes)
	ifelse([$1], , :, [$1])
	;;
    *)
	ifelse([$2], , :, [$2])
	;;
esac
])
dnl
dnl AC_OPT_TIMERS
dnl
AC_DEFUN([AC_OPT_TIMERS], [
AC_ARG_ENABLE(timers,
   AC_HELP_STRING([--enable-timers],
      [Enable performance timers. Values are "yes" or "process", "monotonic", "monotonic_raw" or "raw", "rdtsc", and "no" (default=no)]), [
      AC_DEFINE([WITH_TIMERS], [], [libsff maintains per-module performance timers])
      must_test_gettime=no
      case "$enableval" in
         chrono )
 	    must_test_gettime=no
	    AC_DEFINE([WITH_CHRONO_TIMERS], [], [Using C++-11 chrono feature.])
	    ;;
         yes | process )
 	    must_test_gettime=yes
	    AC_DEFINE([GETTIME_CLOCK_ID], [PER_PROCESS], [Per-process timer.])
	    ;;
         monotonic )
 	    must_test_gettime=yes
	    AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC], [Monotonic timer.])
	    ;;
         monotonic_raw | raw )
 	    must_test_gettime=yes
            AC_CHECK_MONOTONIC_RAW([
 	        AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC_RAW], [Monotonic raw hardware timer.])
            ], [
		AC_MSG_WARN([CLOCK_MONOTONIC_RAW unavailable, using clock_gettime with CLOCK_MONOTONIC.])
 		AC_DEFINE([GETTIME_CLOCK_ID], [MONOTONIC], [Monotonic timer.])
	    ])
	    ;;
 	 rdtsc )
	    AC_CHECK_RDTSC([
	        AC_DEFINE([HAVE_RDTSC], [], [Intel rdtsc instruction])
 		AC_DEFINE([WITH_RDTSC_TIMERS], [], [implementing timers with RDTSC])
	      ], [
		AC_MSG_WARN([RDTSC unavailable, using clock_gettime for performance timers.])
 		must_test_gettime=yes
 	      ])
	    ;;
	 * )
 	    AC_MSG_RESULT([Performance timers are disabled])
	    ;;
      esac
      if test $must_test_gettime = yes; then
	 AC_CHECK_LIB(rt,clock_gettime)
      fi
  ])
])
dnl
dnl AC_OPT_MUTEX_THREADPOOL
dnl
AC_DEFUN([AC_OPT_MUTEX_THREADPOOL], [
AC_ARG_WITH(mutex-threadpool,
   AC_HELP_STRING([--with-mutex-threadpool],
      [Use mutex to control access to threadpool. Values are "yes" or "no" (default=no)]), [
      case "$withval" in
         yes )
	    AC_DEFINE([WITH_MUTEX_THREADPOOL], [], [use mutex to control access to threadpool])
	    ;;
	 * )
 	    AC_MSG_RESULT([using non-blocking queues for threadpool])
	    ;;
      esac
  ])
])

dnl
dnl AC_OPT_WITH_BUILTIN_CAS
dnl
AC_DEFUN([AC_OPT_WITH_BUILTIN_CAS], [
AC_ARG_WITH(built-in-cas,
   AC_HELP_STRING([--with-built-in-cas],
      [Use mutex to control access to threadpool. Values are "yes" or "no" (default=no)]), [
      case "$withval" in
         yes )
	    AC_CHECK_BUILTIN_CAS([
 		AC_DEFINE([WITH_BUILTIN_CAS], [], [using built-in compare-and-swap])
	      ], [
		AC_MSG_WARN([built-in compare-and-swap unavailable, using <atomic>.])
 	      ])
	    ;;
	 * )
 	    AC_MSG_RESULT([using compare-and-swap from <atomic>.])
	    ;;
      esac
  ])
])

dnl
dnl AC_WITH_THREAD_COUNT
dnl
AC_DEFUN([AC_WITH_THREAD_COUNT], [
AC_ARG_WITH(thread-count,
   AC_HELP_STRING([--with-thread-count],
      [Thread-pool size. (default=4)]), [
      AC_DEFINE_UNQUOTED([THREAD_COUNT], $withval, [threadpool size])
  ])
])

dnl
dnl AC_CHECK_BOOST_UNIT_TEST
dnl
AC_DEFUN([AC_CHECK_BOOST_UNIT_TEST], [
   AC_CACHE_CHECK(whether we can compile Boost unit tests,
   ax_cv_boost_unit_test_compile, [
     AC_LANG_PUSH([C++])
     AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[@%:@include <boost/test/unit_test.hpp>]],
     				        [[using boost::unit_test::test_suite;
 				          test_suite* test= BOOST_TEST_SUITE( "Unit test example 1" ); 
 					  return 0;]])], [ax_cv_boost_unit_test_compile=yes])
     AC_LANG_POP([C++])])
if test "x$ax_cv_boost_unit_test_compile" = "xyes"; then
   AC_LANG_PUSH([C++])
   AC_CHECK_LIB([boost_unit_test_framework], [main], [
     AC_DEFINE(HAVE_BOOST_UNIT_TEST_FRAMEWORK,,[define if the Boost::Unit_Test_Framework library is available])
     BOOST_UNIT_TEST_FRAMEWORK_LIB="-lboost_unit_test_framework"
     AC_SUBST(BOOST_UNIT_TEST_FRAMEWORK_LIB)
     ax_cv_boost_unit_test_framework=yes], [ax_cv_boost_unit_test_framework=no])
   AC_LANG_POP([C++])
else
     ax_cv_boost_unit_test_framework=no
fi
if test "x$ax_cv_boost_unit_test_framework" != "xyes"; then
   AC_MSG_WARN([Without the Boost Unit Test Framework you will not be able to run the tests.])
fi
])
