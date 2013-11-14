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

#ifdef WITH_TIMERS

#include <unistd.h>

#include <vector>

#include "timing.h"

using namespace std;

#ifdef WITH_RDTSC_TIMERS

Timer::cycles_t Timer::cycles_per_second() {
    static cycles_t measured_cycles_per_second = 0;

    if (measured_cycles_per_second == 0) {
		// Our goal here is to compute the number of cycles per second on
		// this CPU.  The basic idea is pretty simple: compute the cycles
		// spent a given interval.  For example, compute the cycles in an
		// interval of 1/4 of a second, and then multiply by 4.
		// 
		// However, this simple measurement is affected by a fixed costs
		// in terms of cycles due to whatever it takes to execute
		// usleep(), so here I use TWO measurements and subtract one from
		// the other so as to get rid of those fixed additional costs (as
		// much as possible).
		// 
		cycles_t start = read_cpu_cycles();
		usleep(100000); // sleep for 0.1 second
		cycles_t elapsed1 = read_cpu_cycles() - start;

		start = read_cpu_cycles();
		usleep(350000); // sleep for 0.35 second
		cycles_t elapsed2 = read_cpu_cycles() - start;

		measured_cycles_per_second = (elapsed2 - elapsed1)*4;
    }
    return measured_cycles_per_second;
}

double Timer::read_microseconds () const {
    double res = cycles_total;
    res *= 1000000;
    res /= cycles_per_second();
    return res;
}
#endif

#endif // WITH_TIMERS
