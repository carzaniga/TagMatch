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

#include <vector>

namespace siena {

/** A fine-grained process timer.
 *
 *  This class implements a process timer.  I.e., a timer that
 *  accounts for the execution time of this process.  The
 *  implementation is intended to be as precise as possible, in
 *  absolute terms.
 */
class Timer
{
    unsigned long long usecs;

    unsigned long start_secs;
    unsigned long start_usecs;
public:
    explicit Timer () : usecs(0), start_secs(0), start_usecs(0) {}

    void start ();
    void stop ();
    void reset ();
    unsigned long long read () const;

    static void enable ();
    static void disable ();
    static bool is_enabled ();
};

class TimerStack {
    std::vector<Timer*> stack;
public:
    /** pushes the given timer on this stack of timers.
     *
     *  This class maintains a stack of timers to allow per-procedure
     *  or per-block accounting.  The semantics of the timer stack is
     *  as follows: when a timer T is pushed onto the stack, T is
     *  started.  At the same time, the timer T' that was on top of
     *  the stack (if any) is stopped.  When a timer T is popped from
     *  the stack, the T is stopped, and at the same time the timer T'
     *  that emerges at the top of the stack (if any) is started.
     *  This semantics is illustrated by the following example:
     * 
     *  <code>
     *  TimerStack S;
     *  Timer timer_x;
     *  Timer timer_y;
     *
     *  //...
     *  S.push(timer_x);
     *  //... some code that executes for 3 seconds
     *  S.push(timer_y);
     *  //... some code that executes for 10 seconds
     *  S.pop(timer_y);
     *  //... some code that executes for 2 seconds
     *  S.pop(timer_x);
     *  std::cout << "Tx=" << (timer_x.read() / 1000000) << std::endl;
     *  std::cout << "Ty=" << (timer_y.read() / 1000000) << std::endl;
     *  </code>
     *
     *  the example should output something like:
     *  <code>
     *  Tx=5
     *  Ty=10
     *  </code>
     */
    void push(Timer &);
    void pop();
};

}

#endif
