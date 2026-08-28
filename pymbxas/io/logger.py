#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:41:10 2023

@author: roncoroni
"""

import sys
from datetime import datetime
from io import StringIO

#%%

class Logger(object):
    """
    Custom logger to print to terminal and store the output as a string. Tracks file closure.
    """
    def __init__(self, print_to_terminal=True, log_file=None, append=False,
                 section_context=None):
        self.print_to_terminal = print_to_terminal
        self.log = StringIO()
        self.log_file = log_file
        self.file = None
        self._isclosed = False

        if print_to_terminal:
            self.terminal_write = sys.stdout.write
            self.terminal_flush = sys.stdout.flush
        else:
            self.terminal_write = lambda message: None
            self.terminal_flush = lambda: None

        if log_file:
            if append:
                self.file = open(
                    log_file, 'a', encoding='utf-8', buffering=1)
            else:
                self.file = open(
                    log_file, 'w', encoding='utf-8', buffering=1)

        if section_context is not None:
            self.start_section(section_context)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, message):
        self.terminal_write(message)
        self.log.write(message)
        if self.file and not self._isclosed:
            self.file.write(message)

    def flush(self):
        self.terminal_flush()
        if self.file and not self._isclosed:
            self.file.flush()

    def start_section(self, context):
        """Write a visible, structured boundary before raw backend output."""
        if not hasattr(context, "items"):
            raise TypeError("section_context must be a mapping")
        fields = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **{str(key): value for key, value in context.items()
               if value is not None},
        }
        width = max(len(key) for key in fields)
        separator = "=" * 80
        lines = ["", separator, "BEGIN PyMBXAS SCF"]
        for key, value in fields.items():
            clean_value = str(value).replace("\n", " ")
            lines.append(f"{key:<{width}} : {clean_value}")
        lines.extend((separator, ""))
        self.write("\n".join(lines))

    def close(self):
        if self.file and not self._isclosed:
            self.file.close()
            self._isclosed = True

    def get_log(self):
        return self.log.getvalue()

    def is_closed(self):
        return self._isclosed
