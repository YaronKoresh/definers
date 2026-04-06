from __future__ import annotations

import sys

from definers import constants


def update_system_message(system_message: str) -> None:
    constants.SYSTEM_MESSAGE = system_message
    definers_package = sys.modules.get("definers")
    if definers_package is not None:
        definers_package.SYSTEM_MESSAGE = system_message


def build_system_message(
    name: str | None,
    role: str,
    tone: str | None,
    chattiness: str | None,
    interaction_style: str | None,
    persona_data: dict | None,
    goals: list | None,
    task_rules: list | None,
    output_format: str | None,
    rules: list | None,
    data: list | None,
    verbose: bool,
    friendly: bool,
    formal: bool | None,
    creative: bool | None,
) -> str:
    _ = (rules, verbose, friendly, formal, creative, data)
    parts = [f"You are {role}."]
    if name:
        parts.append(f"Your name is {name}.")
    if tone:
        parts.append(f"Your tone should be {tone}.")
    if chattiness:
        parts.append(f"In terms of verbosity, {chattiness}.")
    if interaction_style:
        parts.append(f"When interacting, {interaction_style}.")
    if persona_data:
        for key, value in persona_data.items():
            parts.append(f"{key} is {value}")
    if goals:
        parts.append("; ".join(goals) + ".")
    if task_rules or output_format:
        parts.append("You must strictly follow these rules:")
        rule_num = 1
        if task_rules:
            for rule in task_rules:
                parts.append(f"{rule_num}. {rule}")
                rule_num += 1
        if output_format:
            parts.append(
                f"{rule_num}. Your final output must be exclusively in the following format: {output_format}."
            )
    return "\n".join(parts)


def set_system_message(
    name: str | None = None,
    role: str = "a helpful AI assistant",
    tone: str | None = None,
    chattiness: str | None = None,
    interaction_style: str | None = None,
    persona_data: dict | None = None,
    goals: list | None = None,
    task_rules: list | None = None,
    output_format: str | None = None,
    rules: list | None = None,
    data: list | None = None,
    verbose: bool = False,
    friendly: bool = True,
    formal: bool | None = None,
    creative: bool | None = None,
) -> None:
    update_system_message(
        build_system_message(
            name,
            role,
            tone,
            chattiness,
            interaction_style,
            persona_data,
            goals,
            task_rules,
            output_format,
            rules,
            data,
            verbose,
            friendly,
            formal,
            creative,
        )
    )
