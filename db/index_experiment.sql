-- Index experiment: does an index help, and does the column order inside it matter?
--
-- Query: collisions per year on 60 mph (97 km/h) roads, 2010-2025.
-- It matches speed_limit exactly (=) and collision_year as a range (between).
-- It returns 265,161 of the 9,116,625 collisions (about 3%).
--
-- Results on this machine:
--   no index                         1,468 ms   Parallel Seq Scan, reads every row
--   index (collision_year, speed_limit)  38 ms   Index Only Scan
--   index (speed_limit, collision_year)  13 ms   Index Only Scan
--
-- Why the order matters: an index is sorted by its first column, then by the
-- second within each value of the first.
--   (speed_limit, collision_year): all 60 mph entries sit together, sorted by year.
--     Postgres jumps to (60, 2010) and reads until (60, 2025). It reads only the
--     265,161 entries it needs.
--   (collision_year, speed_limit): entries for 2010-2025 sit together, but every
--     speed limit is mixed in. Postgres reads all of 2010-2025 (about 2 million
--     entries) and discards the ones that are not 60 mph.
-- Rule: put the column matched with = first, and the column matched as a range second.
--
-- "Index Only Scan ... Heap Fetches: 0" means the answer came from the index alone.
-- The table itself was never read, because the query needs only the two indexed columns.

drop index if exists raw.idx_year_limit;
drop index if exists raw.idx_limit_year;

-- 1. No index
explain analyze
select collision_year, count(*) from raw.collision
where speed_limit = 60 and collision_year between 2010 and 2025
group by collision_year;

-- 2. Range column first
create index idx_year_limit on raw.collision (collision_year, speed_limit);

explain analyze
select collision_year, count(*) from raw.collision
where speed_limit = 60 and collision_year between 2010 and 2025
group by collision_year;

drop index raw.idx_year_limit;

-- 3. Exact-match column first (the winner, left in place)
create index idx_limit_year on raw.collision (speed_limit, collision_year);

explain analyze
select collision_year, count(*) from raw.collision
where speed_limit = 60 and collision_year between 2010 and 2025
group by collision_year;
