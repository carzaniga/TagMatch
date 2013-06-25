//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//  
//  Author: Antonio Carzaniga (firstname.lastname@usi.ch)
//  See the file AUTHORS for full details. 
//  
//  Copyright (C) 2013 Antonio Carzaniga
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
#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>

/** A high-precision timer.
 *
 *  This class implements a high-recision wall-clock timer (so, not a
 *  process-specific timer).  The implementation is intended to be
 *  precise as well as efficient, meaning with minimal overhead.  It
 *  is based on Intel's internal Time Stamp Counter, in particular on
 *  the RDTSCP or RDTSC instructions.
 *
 *  Example:
 *  <code>
 *  Timer t;
 *  // ...
 *  while(not_done) {
 *      // ...
 *      t.start();
 *      // code that we want to measure
 *      t.stop();
 *      // ...
 *  }
 *  cout << "time spent in my code (us): " << t.read_microseconds() << endl;
 */
class Timer
{
public:
    typedef uint64_t cycles_t;

private:
    cycles_t cycles_total;
    cycles_t cycles_start;

#if (defined(HAVE_RDTSCP) || defined(HAVE_RDTSC))
    static inline uint64_t read_cpu_cycles(void) {
	uint32_t lo, hi;
#ifdef HAVE_RDTSCP
	// NOTE: We cannot use "=A", since this would use %rax on
	// x86_64 and return only the lower 32bits of the TSC. RDTSCP
	// is a serializing instruction, meaning that it waits for all
	// issued instructions (and memory accesses) to terminate.
	__asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi));
#else
	// Here we don't have the serializing RDTSCP and instead only
	// have RDTSC, so we must explicitly serialize using a
	// serializing instruction like CPUID.
	__asm__ __volatile__ ("xorl %%eax,%%eax \n        cpuid"
	    ::: "%rax", "%rbx", "%rcx", "%rdx");
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
#endif
	return (uint64_t)hi << 32 | lo;
    }
#else
#error "This code requires Intel's rdtsc instructions"
#endif

public:
    explicit Timer() : cycles_total(0), cycles_start(0) {}

    /** timer calibration.
     *
     *  This method measures the number of cycles per second on this
     *  CPU.  Notice that it pauses the execution for about 0.45
     *  seconds.
     * 
     *  The execution of this method is necessary if one wants to use
     *  timers to read actual time (in microseconds).  In fact, if
     *  it's not called explicitly, it is called automatically once at
     *  the first invocation of read_microseconds().
     */
    static void calibrate();

    /** starts the timer.
     *  
     *  two or more consecutive start() calls are equivalent to the
     *  last call.
     */
    inline void start() {
	cycles_start = read_cpu_cycles();
    }

    /** stops the timer.
     *
     *  if the timer was previously started and not yet stopped, stops
     *  the timer and adds to the total timer value the elapsed
     *  interval between the most recent start() call and this stop()
     *  call.
     */
    inline void stop() {
	cycles_t stop = read_cpu_cycles();
	if (cycles_start != 0) {
	    cycles_total += (stop - cycles_start);
	    cycles_start = 0;
	}
    }

    /** resets the total timer to 0. */
    void reset() {
	cycles_total = 0;
    }

    /** returns the total timer value in cycles. */
    cycles_t read_cycles () const {
	return cycles_total;
    }

    /** returns the total timer value in microseconds. */
    double read_microseconds () const;

    static cycles_t cycles_per_second ();
};

#endif
