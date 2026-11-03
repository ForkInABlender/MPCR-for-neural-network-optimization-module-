#Dylan Kenneth Eliot

"""Menger-sponge-inspired recursive partitions and computational tiles."""
from itertools import product

def menger_cells(depth=1):
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    cells = [(0, 0, 0)]
    for _ in range(depth):
        expanded = []
        for base in cells:
            for offset in product(range(3), repeat=3):
                if sum(v == 1 for v in offset) < 2:
                    expanded.append(tuple(base[i] * 3 + offset[i] for i in range(3)))
        cells = expanded
    return tuple(cells)

def partition_ranges(length, parts):
    if length < 0:
        raise ValueError("length must be non-negative")
    if parts < 1:
        raise ValueError("parts must be positive")
    count = min(length, parts)
    if count == 0:
        return ()
    q, r = divmod(length, count)
    out, start = [], 0
    for i in range(count):
        stop = start + q + (1 if i < r else 0)
        out.append((start, stop))
        start = stop
    return tuple(out)

def sponge_region_order(n, depth=1):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return ()
    cells = menger_cells(depth)
    ordered = sorted(cells, key=lambda p: (sum(p), p[2], p[1], p[0]))
    return tuple(ordered[i % len(ordered)] for i in range(n))

def recursive_ranges(length, max_leaf_size, min_leaf_size=1):
    """Recursively bisect [0,length) until leaves are no larger than max_leaf_size."""
    if length < 0 or max_leaf_size < 1 or min_leaf_size < 1:
        raise ValueError("invalid partition sizes")
    if length == 0:
        return ()
    leaves = []
    def split(start, stop):
        size = stop - start
        if size <= max_leaf_size or size < 2 * min_leaf_size:
            leaves.append((start, stop))
            return
        mid = start + size // 2
        split(start, mid)
        split(mid, stop)
    split(0, length)
    return tuple(leaves)

def sponge_tiles(rows, cols, tile_rows, tile_cols, depth=1):
    """Generate rectangular tiles and deterministic sponge-region metadata."""
    if min(rows, cols) < 0 or min(tile_rows, tile_cols) < 1:
        raise ValueError("invalid dimensions or tile sizes")
    row_ranges = tuple((i, min(i + tile_rows, rows)) for i in range(0, rows, tile_rows))
    col_ranges = tuple((j, min(j + tile_cols, cols)) for j in range(0, cols, tile_cols))
    coords = sponge_region_order(len(row_ranges) * len(col_ranges), depth)
    tiles, k = [], 0
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            tiles.append({
                "row": (r0, r1), "col": (c0, c1),
                "region": coords[k], "ordinal": k
            })
            k += 1
    return tuple(sorted(tiles, key=lambda item: (sum(item["region"]), item["region"])))
