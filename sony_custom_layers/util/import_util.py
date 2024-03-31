# -----------------------------------------------------------------------------
# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
from typing import List, Union

from packaging.requirements import Requirement
from packaging.version import parse
from importlib import metadata


class RequirementError(Exception):
    pass


def validate_pip_requirements(requirements: List[str]):
    """
    Validate that all requirements are installed and meet the version specifications.

    Args:
        requirements: a list of pip-style requirement strings

    Raises:
        RequirementError if any required package is not installed or doesn't meet the version specification
    """
    error = ''
    for req_str in requirements:
        req = Requirement(req_str)
        try:
            installed_ver = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            error += f'\nRequired package {req_str} is not installed'
            continue

        if parse(installed_ver) not in req.specifier:
            error += f'\nRequired {req_str}, installed version {installed_ver}'
    if error:
        raise RequirementError(error)


def is_compatible(requirements: Union[str, List]) -> bool:
    """
    Non-raising requirement(s) check
    Args:
        requirements (str, List): requirement pip-style string or a list of requirement strings

    Returns:
        (bool) whether requirement(s) are satisfied
    """
    requirements = [requirements] if isinstance(requirements, str) else requirements
    try:
        validate_pip_requirements(requirements)
    except RequirementError:
        return False
    return True
