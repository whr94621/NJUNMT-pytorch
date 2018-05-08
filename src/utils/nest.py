def _yield_value(iterable):
    for value in iterable:
      yield value

def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n

def is_sequence(seq):
    return isinstance(seq, (list, tuple))

def flatten(nest):

    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]

def _packed_nest_with_indices(structure, flat, index):

    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(child)
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, tuple(packed) if isinstance(structure, tuple) else packed

def pack_sequence_as(structure, flat_sequence):

    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")

    flat_structure = flatten(structure)

    if len(flat_structure) != len(flat_sequence):
        raise ValueError("Count not pack sequence: expected {0} but got {1}".format(len(flat_structure),
                                                                                    len(flat_structure)))

    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)

    return packed

def _recursive_assert_same_structure(nest1, nest2):
    """Helper function for `assert_same_structure`."""
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            "First structure: %s\n\nSecond structure: %s." % (nest1, nest2))

    if not is_sequence_nest1:
        return  # finished checking

    nest1_as_sequence = [n for n in _yield_value(nest1)]
    nest2_as_sequence = [n for n in _yield_value(nest2)]
    for n1, n2 in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2)


def assert_same_structure(nest1, nest2):

    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("The two structures don't have the same number of "
                         "elements.\n\nFirst structure (%i elements): %s\n\n"
                         "Second structure (%i elements): %s"
                         % (len_nest1, nest1, len_nest2, nest2))
    _recursive_assert_same_structure(nest1, nest2)

def map_structure(func, *structure):

    if not callable(func):
        raise TypeError("func must be callable!")

    if len(structure) == 1 and not is_sequence(structure[0]):
        return func(*structure)

    for other in structure[1:]:
        assert_same_structure(structure[0], other)


    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)

    return pack_sequence_as(
        structure[0], [func(*x) for x in entries])
