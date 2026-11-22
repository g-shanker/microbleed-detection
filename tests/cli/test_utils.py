from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, model_validator

from microbleednet.cli.utils import (
    ConfigField,
    config_fields,
    is_pipe_module,
    load_config,
    parse_config,
)
from microbleednet.errors import ApplicationError, print_application_error
from microbleednet.orchestration.configs import IndexDataConfig
from tests.support_cli_core import FakeConfig


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.toml", None, "Configuration file not found"),
        ("config.json", "{}", "must use the TOML format"),
        ("config", "{}", "<none>"),
        ("config.toml", "not = valid = toml", "Could not load configuration file"),
    ],
)
def test_load_config_rejects_unusable_file(
    tmp_path: Path, name: str, content: str | None, message: str
) -> None:
    path = tmp_path / name
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(ApplicationError, match=message):
        load_config(path)


def test_parse_config_reports_validation_error_as_application_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text("dataset_dir = 'dataset'\n", encoding="utf-8")  # missing fields

    with pytest.raises(ApplicationError) as error:
        parse_config(path, IndexDataConfig, "index-data")

    assert "Configuration validation failed" in str(error.value)
    assert "microbleednet describe index-data" in str(error.value)


def test_parse_config_preserves_nested_application_error(tmp_path: Path) -> None:
    class InvalidConfig(BaseModel):
        @model_validator(mode="after")
        def reject_config(self) -> "InvalidConfig":
            raise ApplicationError(
                category="Manifest",
                summary="Upstream manifest is incomplete",
                fix="Rerun the producing stage",
            )

    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ApplicationError) as error:
        parse_config(path, InvalidConfig, "example")

    assert error.value.category == "Manifest"
    assert error.value.summary == "Upstream manifest is incomplete"
    assert error.value.fix == "Rerun the producing stage"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "category": "Configuration",
                "summary": "Configuration failed",
                "cause": "A field is invalid",
                "fix": "Correct the field",
            },
            "Configuration failed. Cause: A field is invalid. Fix: Correct the field.",
        ),
        ({"category": "Internal", "summary": "Operation failed"}, "Operation failed."),
    ],
)
def test_application_error_string_includes_optional_fields(
    kwargs: dict[str, Any], expected: str
) -> None:
    assert str(ApplicationError(**kwargs)) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "category": "Configuration",
                "summary": "Configuration failed",
                "fix": "Correct the field",
                "context": {"path": "config.toml"},
            },
            "Configuration failed. Fix: Correct the field. "
            "Context: path: config.toml.\n",
        ),
        (
            {"category": "Internal", "summary": "Operation failed"},
            "Operation failed.\n",
        ),
    ],
)
def test_print_application_error_uses_plain_terminal_output(
    capsys: pytest.CaptureFixture[str], kwargs: dict, expected: str
) -> None:
    print_application_error(ApplicationError(**kwargs))

    assert capsys.readouterr().out == expected


def test_config_fields_flatten_nested_models_and_mark_required_fields() -> None:
    assert config_fields(FakeConfig, prefix="") == [
        ConfigField("", "top_level", None, "A top-level field."),
        ConfigField("nested", "knob", "1", "A nested knob."),
        ConfigField("", "trailing", "True", "A trailing top-level field."),
    ]


def test_config_fields_report_literal_options() -> None:
    class Config(BaseModel):
        modality: Literal["T2*-GRE", "SWI", "QSM"] = Field(
            default="T2*-GRE", description="Imaging modality."
        )

    assert config_fields(Config, prefix="") == [
        ConfigField(
            section="",
            key="modality",
            default="'T2*-GRE'",
            description="Imaging modality. Allowed values: 'T2*-GRE', 'SWI', 'QSM'.",
        )
    ]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [({}, False), ({"execute": 1}, False), ({"execute": print}, True)],
)
def test_is_pipe_module_requires_callable_execute(
    attributes: dict, expected: bool
) -> None:
    module = ModuleType("pipe")
    module.__dict__.update(attributes)

    assert is_pipe_module(module) is expected
