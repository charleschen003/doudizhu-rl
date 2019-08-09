import os
import sys

cur_dir, _ = os.path.split(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.insert(0, par_dir)

from server.app import db

db.create_all()
