#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:46:15 2018

@author: ljk
"""

import os
import sys
comment=sys.argv[1]
os.system('git add .')
os.system("git commit -m '"+comment+"'")
os.system('git push origin master')