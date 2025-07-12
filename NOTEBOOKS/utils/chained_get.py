def chained_get(container, path, default=None):
    """Helper function to perform a series of .get() methods on a dictionary
    or return the `default`.

    Parameters
    ----------
    container : dict
        The dictionary on which the .get() methods should be performed.

    path : list or tuple
        The list of keys that should be searched for.

    default : any (optional, default=None)
        The object type that should be returned if the search yields
        no result.
    """
    from functools import reduce

    # Obtain value via reduce
    try:
        return reduce(lambda c, k: c.get(k, default), path, container)
    except (AttributeError, TypeError):
        return default
