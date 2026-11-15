#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glo.data import validate_dataset
root=Path(__file__).resolve().parents[1]/'data/processed'
for experiment in ('two_shapes','three_shapes'):
    print(experiment,validate_dataset(root,experiment))
