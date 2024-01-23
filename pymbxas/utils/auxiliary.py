#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:23:47 2023

@author: roncoroni
"""

# return copy of input as list if not one
def as_list(inp):
    return [inp] if not isinstance(inp, list) else inp.copy()

def s2i(string):
    if string == "beta":
        return 1
    elif string == "alpha":
        return 0
    else:
        raise "ERROR CHANNEL"