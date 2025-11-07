from setuptools import setup, find_packages

setup(
    name="screenalyzer",
    version="0.1",
    packages=find_packages(
        include=["screentime", "screentime.*"],
        exclude=[
            "app", "app.*",
            "api", "api.*",
            "jobs", "jobs.*",
            "logs", "logs.*",
            "data", "data.*",
            "config", "config.*",
            "AGENTS", "AGENTS.*",
            "models", "models.*",
            "assets", "assets.*",
            "harvest", "harvest.*",
            "configs", "configs.*",
            "deprecated", "deprecated.*",
            "diagnostics", "diagnostics.*",
            "tests", "tests.*",
            "scripts", "scripts.*",
            "tools", "tools.*",
            "docs", "docs.*",
        ]
    ),
    python_requires=">=3.8",
)
