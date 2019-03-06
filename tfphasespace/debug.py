#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @file   debug.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   04.03.2019
# =============================================================================
"""Debugging utils."""

import tensorflow as tf


def debug_print(op, msg=''):
    p_op = tf.print(msg, op, summarize=-1)
    with tf.control_dependencies([p_op]):
        op = tf.identity(op)
    return op


# EOF
