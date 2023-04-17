#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Colors:

    """

    Helper class with methods that format strings, and prints them with a given
    colored font in terminals that support ANSI or whatever it's called.

    Usage:

    print(Colors.red("hi"))

    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    END = "\033[00m"

    @classmethod
    def red(cls, s):
        return cls.RED + s + cls.END

    @classmethod
    def green(cls, s):
        return cls.GREEN + s + cls.END

    @classmethod
    def yellow(cls, s):
        return cls.YELLOW + s + cls.END
