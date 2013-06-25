// -*- C++ -*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Authors: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 2002 University of Colorado
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
#ifndef SIENA_ALLOCATOR_H
#define SIENA_ALLOCATOR_H

#include <cstddef>

namespace siena {

/** @brief memory manager used by the forwarding table.
 *
 *  This manager allocates and keeps track of blocks of memory.
 *  Individual segments of these blocks are then made available to
 *  users.  The manager does not keep track of each individual
 *  segment.  Rather it allows the user to deallocate all the memory
 *  at once.  This memory manager is ideal for "dictionary" type data
 *  structures, or other data structures that need to grow
 *  dinamically, but that may be deallocated at once.
 **/
class FTAllocator {
public:
    /** block size for this allocator **/
    static const size_t BSize = 16384;

    /** allocates a memory segment of the given size **/
    void * allocate(size_t s);

    /** recycle all the memory blocks used by this allocator.
     *
     *  this method does not deallocate any block.
     **/
    void recycle();

    /** deallocates all the memory blocks used by this allocator **/
    void clear();

    /** @brief total number of bytes used by this allocator 
     **/
    size_t size() const;

    /** @brief total number of bytes allocated by this allocator 
     **/
    size_t allocated_size() const;

    void attach_malloc_block (void *ptr);
    void attach_sub_allocator (FTAllocator &mem);
    void detach () { master->detach_sub_allocator (*this); }
    FTAllocator &master_allocator() { return *master; }

    FTAllocator();

    ~FTAllocator() { clear (); }

private:
    void detach_sub_allocator (FTAllocator &mem);

    /** memory block **/
    struct block {
	char	bytes[BSize];  // This MUST be the first element in
			       // the structure.
	block *	next;
    };

    struct large_block {
	void	    * ptr;
	large_block * next;
    };

    /** pointer to the list of used blocks **/
    block *		blist;

    /** pointer to the list of available blocks **/
    block *		freeblist;

    /** first available position within the current block **/
    size_t		free_pos;

    /** total number of blocks in use, including those in the free list **/
    size_t		bcount;

    /** bytes allocated in normal blocks **/
    size_t		normal_size;

    /** bytes allocated via malloc **/
    size_t		large_size;

    /** pointer to the list of big blocks **/
    large_block *	largeblist;

    /** pointer to the list of suballocators and to the next allocator **/
    FTAllocator *	suballocs;
    FTAllocator *	master;
    FTAllocator *	next;
};

inline FTAllocator::FTAllocator() 
    : blist(0), freeblist(0), free_pos(0), bcount(0), 
	normal_size(0), large_size(0), largeblist(0), suballocs (0),
	master (0), next(0) { }

} // end namespace siena

inline void * operator new (size_t s, siena::FTAllocator & m) {
    return m.allocate(s);
}

inline void * operator new[] (size_t s, siena::FTAllocator & m) {
    return m.allocate(s);
}

#endif
