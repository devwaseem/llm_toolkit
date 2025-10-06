class TransactionNotStartedError(Exception):
    pass


class TransactionAlreadyRunningError(Exception):
    pass


class TransactionAlreadyFinishedError(Exception):
    pass
