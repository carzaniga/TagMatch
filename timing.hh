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
#ifndef SIENA_TIMING_H
#define SIENA_TIMING_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WITH_TIMERS
#include <time.h>

#ifdef WITH_CHRONO_TIMERS
#include <chrono>
using namespace std::chrono;
#endif

/** A high-precision timer.
 *
 *  This class implements a high-recision wall-clock timer (so, not a
 *  process-specific timer).  This is intended to measure potentially
 *  many but relatively short computations.  More specifically, this
 *  timer is intended to measure things that are not affected by
 *  context-switces.
 *
 *  The implementation is intended to be precise as well as efficient,
 *  meaning with minimal overhead.  It is based on Intel's internal
 *  Time Stamp Counter, in particular on the RDTSCP or RDTSC
 *  instructions.
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
 *  t.reset();
 *  t.start();
 *  // some other code that we want to measure
 *  t.stop();
 *  cout << "microsconds in other code: " << t.read_microseconds() << endl;
 */
class Timer {
#ifdef WITH_CHRONO_TIMERS

	typedef high_resolution_clock clock_type;

	duration<double> total;
	time_point<clock_type> t1;

public:
	Timer(): total() { };

	void reset() {
		total = total.zero();
	}

	inline void start() {
		t1 = clock_type::now();
	}

	inline void stop() {
	    total += clock_type::now() - t1;
	}

	/** returns the total timer value in microseconds. */
	double read_microseconds () const {;
		return duration_cast<microseconds>(total).count();
	}

	/** returns the total timer value in nanoseconds. */
	unsigned long long read_nanoseconds () const {;
		return duration_cast<nanoseconds>(total).count();
	}
#else
#ifdef WITH_RDTSC_TIMERS
private:
	typedef unsigned long long cycles_t;
	cycles_t cycles_total;
	cycles_t cycles_start;
	static cycles_t cycles_per_second ();

public:
	// code taken from http://www.mcs.anl.gov/~kazutomo/rdtsc.html
#if defined(__i386__)
	static inline unsigned long long read_cpu_cycles(void) {
	    unsigned long long int x;
	    asm volatile (".byte 0x0f, 0x31" : "=A" (x));
	    return x;
	}
#elif defined(__x86_64__)
	static inline unsigned long long read_cpu_cycles(void) {
	    unsigned int hi, lo;
	    asm volatile (
			"xorl %%eax, %%eax\n\t"
			"cpuid\n\t"
			"rdtsc"
			: "=a"(lo), "=d"(hi)
			: /* no inputs */
			: "rbx", "rcx");
	    return ((unsigned long long)hi << 32ull) | (unsigned long long)lo;
	}
#endif

public:
	explicit Timer() : cycles_total(0), cycles_start(0) {}

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

	/** returns the total timer value in microseconds. */
	double read_microseconds () const;

#else // no RDTSC, and instead we use clock_gettime

#if (GETTIME_CLOCK_ID == PER_PROCESS)
#define	CLOCK_GETTIME(t) {clock_gettime(CLOCK_PROCESS_CPUTIME_ID, (t));}
#elif (GETTIME_CLOCK_ID == MONOTONIC)
#define	CLOCK_GETTIME(t) {clock_gettime(CLOCK_MONOTONIC, (t));}
#elif (GETTIME_CLOCK_ID == MONOTONIC_RAW)
#define	CLOCK_GETTIME(t) {clock_gettime(CLOCK_MONOTONIC_RAW, (t));}
#else
#error "unknown or undefined GETTIME_CLOCK_ID."
#endif

	struct timespec total;
	struct timespec t1;
    
public:

	static const long NSEC_PER_SEC = 1000000000L;

	Timer() {
	    total.tv_sec = 0;
	    total.tv_nsec = 0;
	};

	void reset() {
	    total.tv_sec = 0;
	    total.tv_nsec = 0;
	}

	inline void start() {
	    CLOCK_GETTIME(&t1);
	}

	inline void stop() {
	    struct timespec t2;

	    CLOCK_GETTIME(&t2);
	    total.tv_sec += (t2.tv_sec - t1.tv_sec);
	    if (t2.tv_nsec >= t1.tv_nsec) {
			total.tv_nsec += (t2.tv_nsec - t1.tv_nsec);
	    } else {
			total.tv_nsec += (NSEC_PER_SEC - t1.tv_nsec + t2.tv_nsec);
			total.tv_sec -= 1;
	    }
	    if (total.tv_nsec >= NSEC_PER_SEC) {
			total.tv_sec += 1;
			total.tv_nsec -= NSEC_PER_SEC;
	    }
	}

	/** returns the total timer value in microseconds. */
	double read_microseconds () const {;
	    unsigned long long nsec = total.tv_sec;
	    nsec *= NSEC_PER_SEC;
	    nsec += total.tv_nsec;
	    return (double)nsec / 1000;
	}

	/** returns the total timer value in nanoseconds. */
	unsigned long long read_nanoseconds () const {;
	    unsigned long long nsec = total.tv_sec;
	    nsec *= NSEC_PER_SEC;
	    nsec += total.tv_nsec;
	    return nsec;
	}
#endif // RDTSC vs. GETTIME
#endif // CHRONO
};
#endif // WITH_TIMERS

#endif
