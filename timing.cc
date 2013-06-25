//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//  
//  Author: Paolo Bonzini <paolo.bonzini@lu.usi.ch>
//  See the file AUTHORS for full details. 
//  
//  Copyright (C) 2005 Paolo Bonzini
//  
//  Siena is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Siena is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
//   
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#include <cassert>
#include <iostream>

#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/resource.h>

#include <vector>
#include <cassert>

#include "timing.h"

using namespace std;

namespace siena {

// High precision timers (low overhead too if we're on x86)
// The x86 timer works best if the system is not loaded.  In this case
// we assume that since the last time SIGPROF fired, we have not had
// any context switch, and sum a precise time given by counting SIGPROF
// signals, and a fractional time given by rdtsc.

static bool enabled;

// We have two main implementations: the first one is the
// high-precision, low-overhead one, which uses a combination of
// scheduler time, computed through a SIGPROF handler, and CPU cycle
// counter (rdtsc on x86 and mftb on ppc).  The second implementation
// is simpler and is based on getrusage(), but because getrusage()
// must go through the kernel, it incurs a high overhead.  So, we use
// the first implementation whenever we have access to the CPU cycle
// counters.  However, if we're profiling the system (see the
// --enable-profiling configure option), we must leave SIGPROF alone.

#if !defined(WITH_PROFILING) && (defined(HAVE_RDTSC) || defined(HAVE_MFTB))

#ifdef HAVE_RDTSC
    typedef unsigned long long cycles_t;
    
    static cycles_t rdtsc() {
        cycles_t x;
        __asm__ volatile ("rdtsc" : "=A" (x));
        return x;
    }
#endif
    
#ifdef HAVE_MFTB
    typedef unsigned long long cycles_t;

    static inline cycles_t rdtsc(void) {
        cycles_t ret;
    
        __asm__ __volatile__("mftb %0" : "=r" (ret) : );

        return ret;
    }
#endif

// most recently updated timer value
static volatile unsigned long cur_secs = 0;
static volatile unsigned long cur_usecs = 0;

static volatile cycles_t last = 0;

static unsigned long usecs_per_timer_interval = 0;
static cycles_t cycles_per_usec = 0;

static void handler (int sig)
{
  cycles_t now = rdtsc ();
  cycles_t cycles_delta = now - last;
  cycles_t curr_freq = cycles_delta / usecs_per_timer_interval;

  /* exponential smoothing: delta = 0.25*diff+0.75*delta */
  if (cycles_per_usec) {
      if (cycles_per_usec > curr_freq) {
	  cycles_per_usec -= (cycles_per_usec - curr_freq) >> 2;
      } else {
	  cycles_per_usec += (curr_freq - cycles_per_usec) >> 2;
      }
  } else {
      cycles_per_usec = curr_freq;
  }

  last = now;
  unsigned long new_usecs = cur_usecs + usecs_per_timer_interval;
  if (new_usecs >= 1000000)
    cur_secs++, new_usecs -= 1000000;

  __asm__ volatile ("" : : : );
  cur_usecs = new_usecs;
}

static void set_timer (unsigned long usecs, bool one_shot)
{
  usecs_per_timer_interval = usecs;

  struct itimerval it;
  it.it_interval.tv_sec = 0;
  it.it_interval.tv_usec = one_shot ? 0 : usecs;
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = usecs;
  signal (SIGPROF, handler);
  setitimer (ITIMER_PROF, &it, 0);
}

static void calibrate () 
{
  // Calibrate the timer during two 20ths of a seconds

  cycles_t prev = last = rdtsc ();

  set_timer (50000, true);
  while (last == prev)
    continue;

  prev = last;
  set_timer (50000, true);
  while (last == prev)
    continue;
}

void Timer::enable () 
{
  if (cycles_per_usec == 0)
    calibrate ();
  if (!enabled)
    set_timer (100000, false); /* 10 Hz */

  enabled = true;
}

void Timer::disable () 
{
  if (enabled)
    set_timer (0, false); /* 10 Hz */

  enabled = false;
}

static void read_current_time (unsigned long &store_secs, unsigned long &store_usecs) {
    unsigned long tmp_usecs, tmp_secs;
    unsigned long latch_usecs = cur_usecs;
    cycles_t diff, tmp_tsc;

    do
	{
	    // To update the data if we read it in the middle of the update.
	    latch_usecs = cur_usecs;
	    __asm__ volatile ("" : : : );
	    tmp_tsc = rdtsc ();
	    diff = tmp_tsc - last;
	    tmp_secs = cur_secs;
	    __asm__ volatile ("" : : : );
	    tmp_usecs = cur_usecs;
	}
    while (tmp_usecs != latch_usecs);

    tmp_usecs += diff / cycles_per_usec;
    while (tmp_usecs >= 1000000)
        tmp_secs++, tmp_usecs -= 1000000;

    store_usecs = tmp_usecs;
    store_secs = tmp_secs;
}

#else

static void read_current_time (unsigned long &store_secs, unsigned long &store_usecs) 
{
  struct rusage ru;

  if (!enabled)
    store_usecs = store_secs = 0;
  else
    {
      getrusage (RUSAGE_SELF, &ru);
      unsigned long tmp_secs = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec;
      unsigned long tmp_usecs = ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;
      if (tmp_usecs >= 1000000)
        tmp_secs++, tmp_usecs -= 1000000;
      store_usecs = tmp_usecs;
      store_secs = tmp_secs;
    }
}

void Timer::enable () 
{
  enabled = true;
}

void Timer::disable () 
{
  enabled = false;
}

#endif

void Timer::reset () 
{
    usecs = 0;
}

void Timer::start () 
{
    if (enabled) 
	read_current_time (start_secs, start_usecs);
}

void Timer::stop () 
{
    if (enabled) {
	unsigned long end_secs, end_usecs;
	read_current_time (end_secs, end_usecs);
	usecs += (end_secs - start_secs) * 1000000LL 
	    + (end_usecs - start_usecs);
	start_secs = start_usecs = 0;
    }
}

unsigned long long Timer::read () const {
    if (start_secs == 0 && start_usecs == 0) {
	// here the timer is stopped, either because we called stop()
	// or because we never called start()
	return usecs;
    } else {
	// here the timer is ``running'' 
	unsigned long secs, usecs;
	read_current_time(secs, usecs);
	return (secs - start_secs) * 1000000LL + (usecs - start_usecs);
    }
}

void TimerStack::push(Timer & x) 
{
    if (!stack.empty()) {
	Timer * t_current = stack.back();
	if (t_current == &x) {
	    stack.push_back(&x);
	    return;
	}
	t_current->stop();
    }
    stack.push_back(&x);
    x.start();
}

void TimerStack::pop() 
{
    Timer * t = stack.back();
    stack.pop_back();
    if (stack.empty()) {
	t->stop();
    } else {
	Timer * t_current = stack.back();
	if (t_current != t) {
	    t->stop();
	    t_current->start();
	}
    }
}

bool Timer::is_enabled () 
{
    return enabled;
}

} //namespace siena
