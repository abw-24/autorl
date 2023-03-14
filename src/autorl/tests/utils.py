
def try_except_assertion_decorator(fn):
    """
    Call an instance method (hence the self), assert that it worked (so that
     the assertion can be picked up by a test runner), add exception as string
     message if needed.
    :param fn: Instance method
    :return: Method wrapped in try except block + an assertion
    """
    def wrapper(self):
        success = True
        try:
            fn(self)
        except Exception as e:
            success = False
            msg = e
        else:
            msg = "Success!"

        assert success, msg

    return wrapper