# Autonomous Chemical Research with LLMs

**Paper Link:** (https://www.nature.com/articles/s41586-023-06792-0)[article]

**Date:** Dec 20, 2023

**Main Idea**
- LLMs - rapid advances in different domains (biological, chemical research)
- Progress in automated chemical research
  - autonomous discovery/optimization of organic reactions, automated flow systems and mobile platforms
  - combination of lab + LLM - autonomous science!
- Coscientist - multi-LLM agent that can autonomously design, plan, and perform complex science experiments
  - connections: internet + documentation, robotic experimentation APIs, other LLMs

**Multi-LLM Architecture**
- Planner - goal of planning based on user input
  - GPT-4 chat-completion instance as assistant
- four main actions:
  - GOOGLE - search internet with web searcher module
    - web searcher is its own LLM with internet connection
  - PYTHON - LLM to perform calculations to prepare experiment with code execution moduloe
  - EXPERIMENT - uses Automation through APIs described by DOCUMENTATION module
    - provides information to main module from the relevant documentation of API
    - currently compatible with *Opentrons Python API* and *Emerald CLoud Lab Symbolic Lab Language*
    - 