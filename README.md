# Swarm-RAG-Multi-Agent-Ollama
The new created multi-agent AI that could be used for different purpose. This AI agents are simply Large Language Models that have been given the ability to interact with outside world.
# Adjustment to Swarm file
if there is a problem within the swarm file with the resulting of non existing of Result module, here's how u fix it:
1. nano ~/.local/lib/python3.10/site-packages/swarm/__init__.py
2. Add this command to the package

from .core import Swarm

from .types import Agent, Response, Result

__all__ = ["Swarm", "Agent", "Response", "Result"]
