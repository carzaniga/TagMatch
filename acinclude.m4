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
