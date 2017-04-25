#ifndef TAGMATCH_HH_INCLUDED
#define TAGMATCH_HH_INCLUDED

#include <vector>

#include "filter.hh"
#include "key.hh"
#include "query.hh"
#include "tagmatch_query.hh"

/** Main interface to the TagMatch system
 *
 *  This class implements the main API into the TagMatch hybrid
 *  CPU/GPU subset matcher.  TagMatch implements what amounts to a
 *  table of Bloom filters (see filter.hh), each associated with a
 *  set of keys.
 *
 *      S_1 -> { k_11, k_12, ... }
 *      S_2 -> { k_21, k_22, ... }
 *      ...
 *
 *  Once all sets are added, and the table is consolidated (see
 *  consolidate()), TagMatch is ready for matching.  TagMatch is
 *  designed for stream processing, to the matching operation does not
 *  immediately return the matching results, but rather uses call-back
 *  mechanism through a match_handler object (see match_handler.hh).
 *  This mechanism can be made synchronous with a simple helper class
 *  synchronous_match_handler (see synchronous_match_handler.hh).
 */
class tagmatch {
public:
	/** Add a tag set and its associated keys to the tag-set table.
	 */
	static void add(const filter_t & S, const std::vector<tagmatch_key_t> & Keys);

	/** Add a tag set and an associated keys to the tag-set table.
	 */
	static void add(const filter_t & S, tagmatch_key_t K);

	/** Remove the association between a tag set (Bloom filter) and a key.
	 *
	 *  Removes K from the keys associated with set S.  If no more
	 *  keys are associated with S, then it also removes S from the
	 *  tagset table.
	 */
	static void remove(const filter_t & set, tagmatch_key_t key);

	/** Consolidate the tagset table for matching.
	 *
	 *  The consolidate prepares the current table for matching.  The
	 *  consolidate process also saves the current state of the tagset
	 *  table so that future calls to add/remove would incrementally
	 *  add and remove tag sets from the current table.
	 */
	static void consolidate();

	/** Consolidate the tagset table for matching.
	 *
	 *  Consolidate the tagset table with a maximal partition size
	 *  max_p, and using the given number of CPU threads.
	 */
	static void consolidate(unsigned int max_p, unsigned int threads);

	/** Return the file name for storing the tagset table.
	 *
	 *  See consolidate().
	 */
	static const char * get_database_filename();

	/** Set the file name for storing the tagset table.
	 *
	 *  See consolidate().
	 */
	static void set_database_filename(const char *);

	/** Start the matching system using a default number of CPU
	 *  threads and GPUs.
	 */
	static void start();

	/** Start the matching system with the given number of CPU threads
	 *  and GPUs.
	 *
	 *  The given number of CPU threads and GPUs become the default
	 *  for any subsequent calls to start().
	 */
	static void start(unsigned int threads, unsigned int gpu_count);

	/** Stop the matching process.
	 */
	static void stop();

	/** Clears the tagset table.
	 */
	static void clear();

	/** Get the latency limit for batch processing of incoming
	 *  matching queries.
	 */
	static unsigned int get_latency_limit_ms();

	/** Set the latency limit for batch processing of incoming
	 *  matching queries.
	 */
	static void set_latency_limit_ms(unsigned int latency_limit);

	/** Match operation: processes one tagset.
	 */
	static void match(tagmatch_query * q, match_handler * h) noexcept;

	/** Match operation: processes one tagset.
	 */
	static void match_unique(tagmatch_query * q, match_handler * h) noexcept;
};

#endif
