# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud chat-completion client wrappers with retry behavior."""

from typing import Any

from tenacity import retry, stop_after_attempt, wait_random_exponential

from coc_labeling.model_clients.openai_client import create_cloud_openai_client


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def completion_with_backoff_cloud(**kwargs: Any) -> Any:
    """Call the configured public endpoint with bounded transient retries."""
    return create_cloud_openai_client().client.chat.completions.create(**kwargs)
