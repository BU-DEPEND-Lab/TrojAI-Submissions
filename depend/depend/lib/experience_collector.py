"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Callable, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr

from depend.core.loggers import Logger

import torch
import torch.nn as nn
from torch_ac.utils.penv import DictList, ParallelEnv
from depend.lib.agent import Agent


 