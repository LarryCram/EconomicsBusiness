def time_run(func):
    """
    wrapper for decorator to time func
    :param func:
    :return:
    """

    import datetime as dt

    def wrapper(*args, **kwargs):
        # Time stamp
        time_now = dt.datetime.now()
        time_start = time_now
        print(f'Run starts at {str(time_now).split(".")[0]}')
        result = func(*args, **kwargs)
        # Time stamp
        time_now = dt.datetime.now()
        duration = time_now - time_start
        minute, second = divmod(duration.total_seconds(), 60)
        print(f'NORMAL TERMINATION at {str(time_now).split(".")[0]} with elapsed time '
              f'{minute} m {second:.2f} s')
        return result

    return wrapper
