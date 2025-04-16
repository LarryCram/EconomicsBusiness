def profile_run(func):
    """
    wrapper for decorator to run profile on func
    :param func:
    :return:
    """

    import cProfile
    import pstats
    from pstats import SortKey

    def wrapper(*args, **kwargs):
        func_name = f"./data/{func.__name__}.pfl"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(func_name)
        p = pstats.Stats(func_name).strip_dirs()
        p.sort_stats(SortKey.CUMULATIVE).print_stats(15)
        return retval

    return wrapper
